# The second copy of the published table.
#
# The stage two table appears twice: once in README.md and once in
# notes/METHODS.md, which is where the long form detail was moved. Two copies of
# the same numbers in two files is exactly the arrangement that drifts, and
# nothing compared them. scripts/check_numbers.py concatenates the two documents
# before searching, so a figure that is right in one file and stale in the other
# still passes it.
#
# This rebuilds the table from results/stage2.csv a fourth time, in Ruby, and
# requires the copy in notes/METHODS.md to match the copy in README.md cell for
# cell as well as matching the data. It also checks the two figures METHODS.md
# quotes in prose and the featuriser accuracy, which lives in run-meta.json and
# is quoted nowhere else.
#
# Ruby 2.6 opens files as US-ASCII, so every read names its encoding.

require "csv"
require "json"

root = ARGV[0] || "."
failures = 0

def read_text(path)
  File.read(path, encoding: "UTF-8")
end

# Table rows, bold markers removed so a bolded number still compares as a number.
def table_rows(path)
  read_text(path).split("\n").map(&:strip)
                 .select { |l| l.start_with?("|") && l.end_with?("|") }
                 .map { |l| l[1..-2].split("|").map { |c| c.gsub("**", "").strip } }
end

def median(v)
  s = v.sort
  s.length.odd? ? s[s.length / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2.0
end

s2 = CSV.read(File.join(root, "results", "stage2.csv"), headers: true)
meta = JSON.parse(read_text(File.join(root, "results", "run-meta.json")))
readme = table_rows(File.join(root, "README.md"))
methods_path = File.join(root, "notes", "METHODS.md")
methods = table_rows(methods_path)
# emphasis stripped so a bolded figure still reads as the figure, and newlines
# collapsed so a sentence wrapped across two lines still reads as one sentence.
methods_text = read_text(methods_path).gsub("**", "").gsub("*", "").gsub(/\s+/, " ")

def med_of(rows, value, model, nfe = nil)
  sel = rows.select { |r| r["model"] == model && (nfe.nil? || r["nfe"] == nfe.to_s) }
  median(sel.map { |r| r[value].to_f })
end

puts "the stage two table, rebuilt in Ruby and compared in both files"
%w[pixel\ DDPM LDM\ f=2 LDM\ f=4 LDM\ f=8].each do |model|
  rows = s2.select { |r| r["model"] == model }
  want = [
    model,
    rows.first["latent_elems"],
    format("%.0f s", med_of(s2, "train_s", model)),
    format("%.2f", med_of(s2, "cfid", model, 10)),
    format("%.2f", med_of(s2, "cfid", model, 25)),
    format("%.2f", med_of(s2, "cfid", model, 50)),
    format("%.4f", med_of(s2, "sw2", model, 50)),
  ]
  in_readme = readme.include?(want)
  in_methods = methods.include?(want)
  failures += 1 unless in_readme && in_methods
  puts format("  %-12s README %s   METHODS.md %s   | %s |",
              model,
              in_readme ? "ok  " : "FAIL",
              in_methods ? "ok  " : "FAIL",
              want.join(" | "))
end

# The two figures METHODS.md quotes in a sentence rather than in a cell.
puts "\nthe figures METHODS.md quotes in prose"
falls_from = format("%.2f", med_of(s2, "cfid", "pixel DDPM", 10))
falls_to = format("%.2f", med_of(s2, "cfid", "pixel DDPM", 50))
sentence = "#{falls_from} to #{falls_to}"
ok = methods_text.include?(sentence)
failures += 1 unless ok
puts format("  %-42s METHODS.md says %-18s %s",
            "the pixel cFID falls across budgets", sentence, ok ? "ok" : "FAIL, not found")

acc = format("%.1f%%", meta["featurizer_acc"] * 100)
ok = methods_text.include?(acc)
failures += 1 unless ok
puts format("  %-42s METHODS.md says %-18s %s",
            "featuriser accuracy, from run-meta.json", acc, ok ? "ok" : "FAIL, not found")

# The element counts the argument about cost rests on.
elems = s2.map { |r| [r["model"], r["latent_elems"]] }.uniq.to_h
ok = elems["pixel DDPM"] == "1024" && elems["LDM f=8"] == "64"
failures += 1 unless ok
puts format("  %-42s %-18s %s", "pixel and f=8 elements per sample",
            "#{elems['pixel DDPM']} and #{elems['LDM f=8']}", ok ? "ok" : "FAIL")

if failures > 0
  puts "\n#{failures} checks failed"
  exit 1
end
puts "\nRuby found the same table in both documents and in the data"
