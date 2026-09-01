// Structural validation of everything under results/, plus a third independent
// rebuild of the two published tables.
//
// results/stage1.csv and results/stage2.csv are the evidence for every number
// in the README. Nothing checked that they are well formed: a truncated write,
// a column that drifted, a duplicated seed row, or a NaN that leaked out of a
// division would all be invisible until someone read the table. This walks
// every tracked results file, then recomputes both README tables from the seed
// level rows and requires every cell to appear in the README, in the right row.
//
// The medians are computed here with a sort and an index, not with the library
// call the Python uses, so a wrong grouping would have to be repeated here to
// survive.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// stage1 is 3 downsampling factors by 3 seeds, stage2 is 4 models by 3 NFE
// budgets by 3 seeds. Both are stated in the README as "median of 3 seeds".
const (
	stage1Rows = 9
	stage2Rows = 36
	seeds      = 3
)

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("only %d rows", len(rows))
	}
	return rows[0], rows[1:], nil
}

func col(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}

// validate reports every structural problem in one file rather than the first,
// so a broken run is diagnosed in one pass.
func validate(path string, wantRows int, keyCols []string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}

	seen := map[string]bool{}
	for _, h := range header {
		if strings.TrimSpace(h) == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}
	if len(rows) != wantRows {
		problems = append(problems, fmt.Sprintf("%d data rows, expected %d", len(rows), wantRows))
	}

	// Every field that parses as a number has to be finite. A column of text
	// such as latent_shape is left alone.
	numeric := make([]bool, len(header))
	for j := range header {
		ok := true
		for _, r := range rows {
			if _, err := strconv.ParseFloat(r[j], 64); err != nil {
				ok = false
				break
			}
		}
		numeric[j] = ok
	}
	for i, r := range rows {
		for j := range header {
			if !numeric[j] {
				continue
			}
			v, _ := strconv.ParseFloat(r[j], 64)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is not finite: %q", i+2, header[j], r[j]))
			}
		}
	}

	// The key columns identify one measurement. Two rows sharing a key means a
	// run was recorded twice, which silently moves a median.
	var idx []int
	for _, k := range keyCols {
		c := col(header, k)
		if c < 0 {
			problems = append(problems, fmt.Sprintf("missing column %q", k))
		}
		idx = append(idx, c)
	}
	if !strings.Contains(strings.Join(problems, ";"), "missing column") {
		keys := map[string]int{}
		for i, r := range rows {
			var parts []string
			for _, c := range idx {
				parts = append(parts, r[c])
			}
			k := strings.Join(parts, "|")
			if prev, dup := keys[k]; dup {
				problems = append(problems,
					fmt.Sprintf("rows %d and %d share the key %s", prev+2, i+2, k))
			}
			keys[k] = i
		}
		// Every key group has to hold the same number of seeds, otherwise
		// "median of 3 seeds" is not what the table shows.
		group := map[string]int{}
		sc := col(header, "seed")
		for _, r := range rows {
			var parts []string
			for _, c := range idx {
				if c != sc {
					parts = append(parts, r[c])
				}
			}
			group[strings.Join(parts, "|")]++
		}
		for k, n := range group {
			if n != seeds {
				problems = append(problems, fmt.Sprintf("group %s has %d seeds, expected %d", k, n, seeds))
			}
		}
	}
	return problems
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

type table struct {
	header []string
	rows   [][]string
}

func load(path string) (table, error) {
	h, r, err := readCSV(path)
	return table{h, r}, err
}

func (t table) num(row int, name string) float64 {
	v, err := strconv.ParseFloat(t.rows[row][col(t.header, name)], 64)
	if err != nil {
		panic(fmt.Sprintf("%s is not a number: %v", name, err))
	}
	return v
}

func (t table) str(row int, name string) string { return t.rows[row][col(t.header, name)] }

// groupBy collects one numeric column, keyed by the string values of others.
func (t table) groupBy(keys []string, value string) map[string][]float64 {
	out := map[string][]float64{}
	for i := range t.rows {
		var parts []string
		for _, k := range keys {
			parts = append(parts, t.str(i, k))
		}
		key := strings.Join(parts, "|")
		out[key] = append(out[key], t.num(i, value))
	}
	return out
}

// readmeRows returns every markdown table row in the README as a cell slice,
// with bold markers removed so a bolded number still compares as a number.
func readmeRows(path string) ([][]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out [][]string
	for _, line := range strings.Split(string(b), "\n") {
		s := strings.TrimSpace(line)
		if !strings.HasPrefix(s, "|") || !strings.HasSuffix(s, "|") {
			continue
		}
		var cells []string
		for _, c := range strings.Split(strings.Trim(s, "|"), "|") {
			cells = append(cells, strings.TrimSpace(strings.ReplaceAll(c, "**", "")))
		}
		out = append(out, cells)
	}
	return out, nil
}

func hasRow(rows [][]string, want []string) bool {
	for _, r := range rows {
		if len(r) != len(want) {
			continue
		}
		same := true
		for i := range r {
			if r[i] != want[i] {
				same = false
				break
			}
		}
		if same {
			return true
		}
	}
	return false
}

func main() {
	root := flag.String("root", "..", "repository root")
	flag.Parse()

	results := filepath.Join(*root, "results")
	failures := 0

	// 1. structure
	for _, spec := range []struct {
		name string
		rows int
		keys []string
	}{
		{"stage1.csv", stage1Rows, []string{"f", "seed"}},
		{"stage2.csv", stage2Rows, []string{"model", "nfe", "seed"}},
	} {
		p := filepath.Join(results, spec.name)
		probs := validate(p, spec.rows, spec.keys)
		if len(probs) == 0 {
			fmt.Printf("results/%-12s well formed, %d rows, no duplicate key, all finite\n",
				spec.name, spec.rows)
		} else {
			failures++
			fmt.Printf("results/%s FAILED\n", spec.name)
			for _, pr := range probs {
				fmt.Printf("    %s\n", pr)
			}
		}
	}

	// 2. run-meta.json has to describe the run the CSVs record
	metaOK := true
	var meta struct {
		Seeds []int `json:"seeds"`
		Fs    []int `json:"fs"`
		NFEs  []int `json:"nfes"`
	}
	if b, err := os.ReadFile(filepath.Join(results, "run-meta.json")); err != nil {
		fmt.Printf("results/run-meta.json unreadable: %v\n", err)
		metaOK = false
	} else if err := json.Unmarshal(b, &meta); err != nil {
		fmt.Printf("results/run-meta.json is not valid JSON: %v\n", err)
		metaOK = false
	}
	s1, err1 := load(filepath.Join(results, "stage1.csv"))
	s2, err2 := load(filepath.Join(results, "stage2.csv"))
	if err1 != nil || err2 != nil {
		fmt.Printf("cannot read the results tables: %v %v\n", err1, err2)
		os.Exit(1)
	}
	if metaOK {
		seen := map[string]bool{}
		for i := range s1.rows {
			seen["f"+s1.str(i, "f")] = true
			seen["seed"+s1.str(i, "seed")] = true
		}
		for i := range s2.rows {
			seen["nfe"+s2.str(i, "nfe")] = true
		}
		var missing []string
		for _, f := range meta.Fs {
			if !seen[fmt.Sprintf("f%d", f)] {
				missing = append(missing, fmt.Sprintf("f=%d", f))
			}
		}
		for _, s := range meta.Seeds {
			if !seen[fmt.Sprintf("seed%d", s)] {
				missing = append(missing, fmt.Sprintf("seed=%d", s))
			}
		}
		for _, n := range meta.NFEs {
			if !seen[fmt.Sprintf("nfe%d", n)] {
				missing = append(missing, fmt.Sprintf("nfe=%d", n))
			}
		}
		if len(missing) == 0 {
			fmt.Printf("run-meta.json      %d seeds, %d factors, %d budgets, all present in the CSVs\n",
				len(meta.Seeds), len(meta.Fs), len(meta.NFEs))
		} else {
			failures++
			fmt.Printf("run-meta.json FAILED, not in the CSVs: %s\n", strings.Join(missing, ", "))
		}
	} else {
		failures++
	}

	// 3. rebuild both README tables from the seed level rows
	rr, err := readmeRows(filepath.Join(*root, "README.md"))
	if err != nil {
		fmt.Printf("cannot read README.md: %v\n", err)
		os.Exit(1)
	}

	psnr := s1.groupBy([]string{"f"}, "psnr")
	rfid := s1.groupBy([]string{"f"}, "rfid")
	comp := s1.groupBy([]string{"f"}, "compression")
	wall := s1.groupBy([]string{"f"}, "wall_s")
	shape := map[string]string{}
	for i := range s1.rows {
		shape[s1.str(i, "f")] = s1.str(i, "latent_shape")
	}
	bad := 0
	for _, f := range []string{"2", "4", "8"} {
		lo, hi := math.Inf(1), math.Inf(-1)
		for _, v := range rfid[f] {
			lo, hi = math.Min(lo, v), math.Max(hi, v)
		}
		want := []string{
			f,
			shape[f],
			fmt.Sprintf("%.1fx", median(comp[f])),
			fmt.Sprintf("%.2f dB", median(psnr[f])),
			fmt.Sprintf("%.3f", median(rfid[f])),
			fmt.Sprintf("%.3f to %.3f", lo, hi),
			fmt.Sprintf("%.0f s", median(wall[f])),
		}
		if !hasRow(rr, want) {
			bad++
			fmt.Printf("    stage one f=%s not in the README as | %s |\n", f, strings.Join(want, " | "))
		}
	}
	if bad == 0 {
		fmt.Printf("stage one table    3 rows, 21 cells, rebuilt from the 9 seed rows and found in the README\n")
	} else {
		failures++
	}

	cfid := s2.groupBy([]string{"model", "nfe"}, "cfid")
	sw2 := s2.groupBy([]string{"model", "nfe"}, "sw2")
	train := s2.groupBy([]string{"model"}, "train_s")
	elems := map[string]string{}
	for i := range s2.rows {
		elems[s2.str(i, "model")] = s2.str(i, "latent_elems")
	}
	bad = 0
	for _, m := range []string{"pixel DDPM", "LDM f=2", "LDM f=4", "LDM f=8"} {
		want := []string{
			m,
			elems[m],
			fmt.Sprintf("%.0f s", median(train[m])),
			fmt.Sprintf("%.2f", median(cfid[m+"|10"])),
			fmt.Sprintf("%.2f", median(cfid[m+"|25"])),
			fmt.Sprintf("%.2f", median(cfid[m+"|50"])),
			fmt.Sprintf("%.4f", median(sw2[m+"|50"])),
		}
		if !hasRow(rr, want) {
			bad++
			fmt.Printf("    stage two %s not in the README as | %s |\n", m, strings.Join(want, " | "))
		}
	}
	if bad == 0 {
		fmt.Printf("stage two table    4 rows, 28 cells, rebuilt from the 36 seed rows and found in the README\n")
	} else {
		failures++
	}

	if failures > 0 {
		fmt.Printf("\n%d checks failed\n", failures)
		os.Exit(1)
	}
	fmt.Printf("\nGo validated both results files and rebuilt all 49 published table cells\n")
}
