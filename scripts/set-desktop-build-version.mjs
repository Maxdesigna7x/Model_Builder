// Give every GitHub Actions build an increasing installer version, including on Windows.
import { readFileSync, writeFileSync } from 'node:fs';

const filename = process.argv[2] || 'src-tauri/tauri.conf.json';
const run = process.env.GITHUB_RUN_NUMBER || '';
if (!/^[1-9]\d*$/.test(run) || Number(run) > 65535) {
  throw new Error('GITHUB_RUN_NUMBER must be an integer between 1 and 65535.');
}
const config = JSON.parse(readFileSync(filename, 'utf8'));
const match = /^(\d+)\.(\d+)\.\d+$/.exec(config.version);
if (!match) throw new Error('Expected a numeric major.minor.patch Tauri version.');
config.version = `${match[1]}.${match[2]}.${run}`;
writeFileSync(filename, JSON.stringify(config, null, 2) + '\n');
console.log(`Desktop installer version: ${config.version}`);
