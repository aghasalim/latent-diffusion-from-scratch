// The headline sentence, recomputed from results/stage2.csv.
//
// The first thing anyone reads is "5.3x faster" and "6.2x better". Those
// multiples are in no results file. They were divided by hand once, and
// scripts/check_numbers.py explicitly does not cover claims written in words,
// so the two numbers with the most reach in the repository were the two nothing
// checked. This divides the published medians again and requires the sentence
// in the README to be the sentence the division produces.
//
// It also checks the two compression factors the headline names and the wall
// clock the intro rounds off, both of which come from files rather than prose.
//
// Run: node verify/ratios.mjs <root>

import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.argv[2] ?? ".";

function readCSV(path) {
  // the results CSVs are written with CRLF line endings, so strip the \r
  const lines = readFileSync(path, "utf8").replace(/\r/g, "").trim().split("\n");
  // quoted fields appear in stage1 (latent_shape), so split on commas outside quotes
  const split = (line) => line.match(/("[^"]*"|[^,]+)/g).map((c) => c.replace(/^"|"$/g, ""));
  const header = split(lines[0]);
  return lines.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [header[i], v])));
}

const median = (v) => {
  const s = [...v].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

const s1 = readCSV(join(root, "results", "stage1.csv"));
const s2 = readCSV(join(root, "results", "stage2.csv"));
const meta = JSON.parse(readFileSync(join(root, "results", "run-meta.json"), "utf8"));
const readme = readFileSync(join(root, "README.md"), "utf8");

const med = (rows, pick, value) => median(rows.filter(pick).map((r) => Number(r[value])));

let failures = 0;
// Every expected string below is built from the number just computed, so a
// changed CSV changes the string this looks for and the README stops matching.
function says(what, got, text) {
  const ok = readme.includes(text);
  if (!ok) failures++;
  console.log(
    `  ${what.padEnd(44)} ${got.padEnd(12)} README says ${text.padEnd(16)} ${ok ? "ok" : "FAIL, not in the README"}`,
  );
}

const trainPixel = med(s2, (r) => r.model === "pixel DDPM", "train_s");
const cfidPixel = med(s2, (r) => r.model === "pixel DDPM" && r.nfe === "50", "cfid");

console.log("the headline, from the medians of results/stage2.csv");
for (const model of ["LDM f=4", "LDM f=8"]) {
  const t = med(s2, (r) => r.model === model, "train_s");
  const c = med(s2, (r) => r.model === model && r.nfe === "50", "cfid");
  const speed = trainPixel / t;
  const quality = cfidPixel / c;
  says(`${model} trains faster than pixel by`, `${speed.toFixed(4)}x`,
       `${speed.toFixed(1)}x faster`);
  says(`${model} cFID at 50 NFE is better by`, `${quality.toFixed(4)}x`,
       `${quality.toFixed(1)}x better`);
}

// The compressions the headline names, from stage one rather than from prose.
console.log("\nthe compression factors the headline names");
for (const [f, tail] of [[4, "compressed"], [8, "compression"]]) {
  const c = med(s1, (r) => Number(r.f) === f, "compression");
  const n = Number.isInteger(c) ? c.toFixed(0) : c.toFixed(1);
  says(`f=${f} compression`, `${c.toFixed(1)}x`, `${n}x ${tail}`);
}

// The intro rounds the wall clock. Check the rounding, not the raw seconds.
console.log("\nthe wall clock the intro rounds off");
const minutes = meta.wall_clock_s / 60;
const rounded = Math.round(minutes / 10) * 10;
says("total run time", `${minutes.toFixed(1)} min`, `about ${rounded} minutes`);

// Seed count, which every "median of 3 seeds" line depends on.
const nseeds = new Set(s2.map((r) => r.seed)).size;
const seedsOK = nseeds === meta.seeds.length && readme.includes("Median of 3 seeds");
if (!seedsOK) failures++;
console.log(
  `  ${"distinct seeds in stage2.csv".padEnd(44)} ${String(nseeds).padEnd(12)} README says ${"Median of 3 seeds".padEnd(16)} ${seedsOK ? "ok" : "FAIL"}`,
);

if (failures > 0) {
  console.log(`\n${failures} checks failed`);
  process.exit(1);
}
console.log("\nJavaScript reproduces every multiple in the headline sentence");
