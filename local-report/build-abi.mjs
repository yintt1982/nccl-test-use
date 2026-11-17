import { mkdir, copyFile, writeFile, rm, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';

const runFile = promisify(execFile);
const output = resolve(process.argv[2] || 'local-report/build');
const root = resolve('local-report/abi');
const vswhere = resolve('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe');
const { stdout: installation } = await runFile(vswhere, ['-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath']);
if (!installation.trim()) throw Error('Visual Studio C++ tools were not found');
const visualStudio = installation.trim();
const toolsets = await readdir(join(visualStudio, 'VC/Tools/MSVC'));
const toolset = toolsets.sort((a, b) => b.localeCompare(a, undefined, { numeric: true }))[0];
const cl = join(visualStudio, 'VC/Tools/MSVC', toolset, 'bin/Hostx64/x64/cl.exe');
const environmentBatch = join(visualStudio, 'VC/Auxiliary/Build/vcvars64.bat');
const environmentScript = join(output, 'compiler-environment.cmd');
await writeFile(environmentScript, '@call "' + environmentBatch + '" >nul\r\n@set\r\n');
const { stdout: environmentOutput } = await runFile('cmd.exe', ['/d', '/c', environmentScript], { maxBuffer: 4_000_000 });
await rm(environmentScript, { force: true });
const compilerEnvironment = { ...process.env };
for (const line of environmentOutput.split(/\r?\n/)) {
  const separator = line.indexOf('=');
  if (separator > 0) compilerEnvironment[line.slice(0, separator)] = line.slice(separator + 1);
}
await mkdir(output, { recursive: true });
await copyFile(resolve('local-report/fixture.json'), join(output, 'fixture.json'));
async function compile(args) {
  const command = '"' + cl + '" ' + args.map(arg => '"' + arg + '"').join(' ');
  const result = spawn(cl, args, { stdio: 'inherit', env: compilerEnvironment });
  const code = await new Promise(resolve => result.on('close', resolve));
  if (code !== 0) throw Error('MSVC failed with exit code ' + code);
}
await compile(['/nologo', '/EHsc', '/std:c++17', '/I', root, '/I', resolve('nccl-tests-master/src'), join(root, 'profiler_plugin.cpp'), '/LD', '/link', '/OUT:' + join(output, 'profiler-plugin.dll')]);
await compile(['/nologo', '/EHsc', '/std:c++17', '/I', root, '/I', resolve('nccl-tests-master/src'), join(root, 'profiler_host.cpp'), '/link', '/OUT:' + join(output, 'profiler-host.exe')]);
