# Moku API Documentation VuePress site

This repo contains the documentation + examples for the Moku API, Moku Cloud Compile, and Moku CLI.

## Getting Started

1. Download Node.js (<https://nodejs.org/en/download/>)
2. Clone this repository
3. run "npm install"
4. run "npm run-script docs:dev". Follow the log and grab the url where it is serving the html.
5. To build, run "npm run-script docs:build". This spits out the static HTML for hosting.

## Markdown Linting

Basic markdown linting has been enabled to catch common spacing. Currently manually linting via the VS Code extension. Linting should be setup via ESlint and run on build.

-   [VS Code Plugin](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint)
-   [Rules](https://github.com/DavidAnson/markdownlint/tree/v0.34.0/doc)

### Broken links

To catch broken links between pages please enable Markdown Validation in your VS code setup.

-   navigate to preferences: Open Settings (UI)
-   search for `markdown validate`
-   Markdown > Validate: Enabled (the default is Disabled)

## MCC Examples

MCC Examples are linked via git submodules. Original files are [stored here](https://gitlab.com/liquidinstruments/cloud-compile/examples)

## To Do

-   Setup automatic markdown linting on build

## Questions

-   Do we want all examples in a sinple public repo?
