// Copyright (C) 2026 Chanwit Kaewkasi
// SPDX-License-Identifier: MIT

// Command brief is the CLI entrypoint. All logic lives in cmd/ and
// internal/engine/; this file only wires the two together.
package main

import "github.com/chanwit/brief/cmd"

func main() {
	cmd.Execute()
}
