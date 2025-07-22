import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import os from 'os';
import sessionManager from './sessionManager.js';

let activeOllamaProcesses = new Map(); // Track active processes by session ID

async function spawnOllama(command, options = {}, ws) {
  return new Promise(async (resolve, reject) => {
    const { sessionId, projectPath, cwd, resume, toolsSettings, permissionMode, images, model } = options;
    let capturedSessionId = sessionId; // Track session ID throughout the process
    let sessionCreatedSent = false; // Track if we've already sent session-created event
    let fullResponse = ''; // Accumulate the full response

    const args = ['run', model];

    // Add tool use flags for supported models
    if (toolsSettings.useTools) {
        if (['llama2', 'codellama'].includes(model)) {
            args.push('--tool-use');
        }
    }

    // Add mcp flag for supported models
    if (toolsSettings.useMcp) {
        if (['llama2', 'codellama'].includes(model)) {
            args.push('--mcp');
        }
    }

    if (command && command.trim()) {
      if (sessionId) {
        const context = sessionManager.buildConversationContext(sessionId);
        if (context) {
          const fullPrompt = context + command;
          args.push(fullPrompt);
        } else {
          args.push(command);
        }
      } else {
        args.push(command);
      }
    }

    const workingDir = (cwd || process.cwd()).replace(/[^\x20-\x7E]/g, '').trim();

    const tempImagePaths = [];
    let tempDir = null;
    if (images && images.length > 0) {
      try {
        tempDir = path.join(workingDir, '.tmp', 'images', Date.now().toString());
        await fs.mkdir(tempDir, { recursive: true });

        for (const [index, image] of images.entries()) {
          const matches = image.data.match(/^data:([^;]+);base64,(.+)$/);
          if (!matches) {
            continue;
          }

          const [, mimeType, base64Data] = matches;
          const extension = mimeType.split('/')[1] || 'png';
          const filename = `image_${index}.${extension}`;
          const filepath = path.join(tempDir, filename);

          await fs.writeFile(filepath, Buffer.from(base64Data, 'base64'));
          tempImagePaths.push(filepath);
        }

        if (tempImagePaths.length > 0 && command && command.trim()) {
          const imageNote = `\n\n[Images attached: ${tempImagePaths.length} image(s) are available at the following paths:]\n${tempImagePaths.map((p, i) => `${i + 1}. ${p}`).join('\n')}`;
          const modifiedCommand = command + imageNote;

          const promptIndex = args.indexOf(command);
          if (promptIndex !== -1) {
            args[promptIndex] = modifiedCommand;
          }
        }
      } catch (error) {
        // console.error('Error processing images for Ollama:', error);
      }
    }

    if (options.debug) {
      args.push('--verbose');
    }

    const ollamaPath = process.env.OLLAMA_PATH || 'ollama';

    const ollamaProcess = spawn(ollamaPath, args, {
      cwd: workingDir,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env }
    });

    ollamaProcess.tempImagePaths = tempImagePaths;
    ollamaProcess.tempDir = tempDir;

    const processKey = capturedSessionId || sessionId || Date.now().toString();
    activeOllamaProcesses.set(processKey, ollamaProcess);

    ollamaProcess.sessionId = processKey;

    ollamaProcess.stdin.end();

    let hasReceivedOutput = false;
    const timeoutMs = 30000; // 30 seconds
    const timeout = setTimeout(() => {
      if (!hasReceivedOutput) {
        ws.send(JSON.stringify({
          type: 'ollama-error',
          error: 'Ollama CLI timeout - no response received'
        }));
        ollamaProcess.kill('SIGTERM');
      }
    }, timeoutMs);

    if (command && capturedSessionId) {
      sessionManager.addMessage(capturedSessionId, 'user', command);
    }

    let outputBuffer = '';

    ollamaProcess.stdout.on('data', (data) => {
      const rawOutput = data.toString();
      outputBuffer += rawOutput;
      hasReceivedOutput = true;
      clearTimeout(timeout);

      if (rawOutput) {
        fullResponse += rawOutput;
        ws.send(JSON.stringify({
          type: 'ollama-response',
          data: {
            type: 'message',
            content: rawOutput
          }
        }));
      }

      if (!sessionId && !sessionCreatedSent && !capturedSessionId) {
        capturedSessionId = `ollama_${Date.now()}`;
        sessionCreatedSent = true;

        sessionManager.createSession(capturedSessionId, cwd || process.cwd());

        if (command) {
          sessionManager.addMessage(capturedSessionId, 'user', command);
        }

        if (processKey !== capturedSessionId) {
          activeOllamaProcesses.delete(processKey);
          activeOllamaProcesses.set(capturedSessionId, ollamaProcess);
        }

        ws.send(JSON.stringify({
          type: 'session-created',
          sessionId: capturedSessionId
        }));
      }
    });

    ollamaProcess.stderr.on('data', (data) => {
      const errorMsg = data.toString();
      ws.send(JSON.stringify({
        type: 'ollama-error',
        error: errorMsg
      }));
    });

    ollamaProcess.on('close', async (code) => {
      clearTimeout(timeout);

      const finalSessionId = capturedSessionId || sessionId || processKey;
      activeOllamaProcesses.delete(finalSessionId);

      if (finalSessionId && fullResponse) {
        sessionManager.addMessage(finalSessionId, 'assistant', fullResponse);
      }

      ws.send(JSON.stringify({
        type: 'ollama-complete',
        exitCode: code,
        isNewSession: !sessionId && !!command
      }));

      if (ollamaProcess.tempImagePaths && ollamaProcess.tempImagePaths.length > 0) {
        for (const imagePath of ollamaProcess.tempImagePaths) {
          await fs.unlink(imagePath).catch(err => {});
        }
        if (ollamaProcess.tempDir) {
          await fs.rm(ollamaProcess.tempDir, { recursive: true, force: true }).catch(err => {});
        }
      }

      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Ollama CLI exited with code ${code}`));
      }
    });

    ollamaProcess.on('error', (error) => {
      const finalSessionId = capturedSessionId || sessionId || processKey;
      activeOllamaProcesses.delete(finalSessionId);

      ws.send(JSON.stringify({
        type: 'ollama-error',
        error: error.message
      }));

      reject(error);
    });
  });
}

function abortOllamaSession(sessionId) {
  let process = activeOllamaProcesses.get(sessionId);
  let processKey = sessionId;

  if (!process) {
    for (const [key, proc] of activeOllamaProcesses.entries()) {
      if (key.includes(sessionId) || sessionId.includes(key)) {
        process = proc;
        processKey = key;
        break;
      }
    }
  }

  if (process) {
    try {
      process.kill('SIGTERM');

      setTimeout(() => {
        if (activeOllamaProcesses.has(processKey)) {
          try {
            process.kill('SIGKILL');
          } catch (e) {
          }
        }
      }, 2000);

      activeOllamaProcesses.delete(processKey);
      return true;
    } catch (error) {
      activeOllamaProcesses.delete(processKey);
      return false;
    }
  }

  return false;
}

export {
  spawnOllama,
  abortOllamaSession
};
