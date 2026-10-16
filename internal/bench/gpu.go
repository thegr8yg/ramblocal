package bench

import (
	"os/exec"
	"strconv"
	"strings"
)

// QueryGPU reads the first GPU's name and used memory from nvidia-smi.
func QueryGPU() (string, int, error) {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return "", 0, err
	}
	line := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)[0]
	parts := strings.Split(line, ",")
	if len(parts) != 2 {
		return "", 0, strconv.ErrSyntax
	}
	used, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	return strings.TrimSpace(parts[0]), used, err
}
