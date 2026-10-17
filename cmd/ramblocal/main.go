// Command ramblocal is the local dictation tool. Rung 1 provides `transcribe` and `doctor`.
package main

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/thegr8yg/ramblocal/internal/version"
)

const usage = `usage: ramblocal <command> [flags]

commands:
  transcribe <file.wav>   transcribe a 16 kHz WAV file to text
  doctor                  check the model, ONNX Runtime and GPU setup
  dictate [--key f9]      hold a key, speak, release: the text is pasted (Windows; see dictate -h)
  keys                    show the key names the keyboard reports (for choosing --key)
  version                 print the version
`

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		if !errors.Is(err, errUsage) {
			fmt.Fprintln(os.Stderr, "ramblocal:", err)
		}
		os.Exit(1)
	}
}

var errUsage = errors.New("usage")

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return errUsage
	}
	switch args[0] {
	case "transcribe":
		return runTranscribe(args[1:], stdout, stderr)
	case "doctor":
		return runDoctor(args[1:], stdout, stderr)
	case "dictate":
		return runDictate(args[1:], stderr)
	case "keys":
		return runKeys(args[1:], stdout, stderr)
	case "version":
		fmt.Fprintln(stdout, version.Version)
		return nil
	case "-h", "--help", "help":
		fmt.Fprint(stdout, usage)
		return nil
	}
	fmt.Fprint(stderr, usage)
	return fmt.Errorf("unknown command %q", args[0])
}
