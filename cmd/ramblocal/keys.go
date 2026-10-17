package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
)

// probeKeys prints the key names the keyboard reports, for choosing a --key value.
// TODO: set probeKeys = hotkey.Probe once internal/win/hotkey provides Probe(ctx, io.Writer) error.
var probeKeys func(ctx context.Context, w io.Writer) error

const keysUsage = `usage: ramblocal keys

Prints the key names the keyboard reports while you press keys, for choosing a dictate --key value.
Fn is usually invisible to Windows, so it may not appear here.
`

func runKeys(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("keys", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, keysUsage) }
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("keys takes no arguments (got %q)", fs.Args())
	}
	if probeKeys == nil {
		return errors.New("keys: the keyboard probe is not connected in this build")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	return probeKeys(ctx, stdout)
}
