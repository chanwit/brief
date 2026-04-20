BINARY  := rag-engine
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w

GOOS    ?= $(shell go env GOOS)
GOARCH  ?= $(shell go env GOARCH)

DIST    := dist
STAGE   := $(DIST)/$(BINARY)-$(VERSION)-$(GOOS)-$(GOARCH)
ARCHIVE := $(STAGE).tar.gz

.PHONY: all build test vet tidy clean dist

all: build

build:
	go build -trimpath -ldflags "$(LDFLAGS)" -o $(BINARY) .

test:
	go test ./...

vet:
	go vet ./...

tidy:
	go mod tidy

clean:
	rm -rf $(DIST) $(BINARY)

# Build and package a release tarball for the current GOOS/GOARCH.
# Because the ONNX runtime binding uses cgo, this target is host-native —
# CI produces a separate archive on each OS runner.
dist:
	@mkdir -p $(STAGE)
	GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -trimpath -ldflags "$(LDFLAGS)" -o $(STAGE)/$(BINARY) .
	tar -C $(DIST) -czf $(ARCHIVE) $(notdir $(STAGE))
	rm -rf $(STAGE)
	@echo "Packaged $(ARCHIVE)"
