# Moku API Documentation VuePress site

This repo contains the documentation + examples for the Moku API, Moku Compile, Moku Neural Network, and Moku CLI.

## Repo Vitals

Maintainer is @ben.nizette.

Branch off `next`.

Submissions are by Merge Request targeting `next`. Assignee must be the maintainer, extra reviewers are allowed if necessary. Merge Requests must be fast-forward merges at the time the MR is created.

Release is by _continuous deployment_. The `next` branch is staged [here](https://next.documentation.liquidinstruments.com/). The `master` branch is deployed [here](https://apis.liquidinstruments.com).

Jira tags in commits are recommended but not required. Multi-line commit messages are recommended but not required.

## Getting Started

1. Download [Node.js](https://nodejs.org/en/download/) Version >= 16. If you intend to develop more than one Node-based project, installation by [nvm](https://github.com/nvm-sh/nvm) is recommended.
2. Clone this repository and install submodules
3. run `npm install`
4. run `npm run docs:dev`. Follow the log and grab the url where it is serving the html. This development site will automatically refresh when the source files change.
5. To build, run `npm run docs:build`. This spits out the static HTML for hosting. The actual deployment is done by a CD pipeline however it's a good idea to run this command locally before creating a Merge Request to catch issues in the static rendering process.

### Building with Node.js >= 18

If you are using a version of Node.js >= version 18 and are seeing errors when trying to build the static HTML, install cross-env with `npm install --save-dev cross-env` and build with `npm run docs:build:node18`

### Testing the build output locally

You can use the [http-server](https://www.npmjs.com/package/http-server) package to view and test your output.

-   Run `npx http-server`
-   See the library for full options

## Examples

This documentation site contains copies of several example scripts. These examples are linked via git submodule to the [moku examples](https://github.com/liquidinstruments/moku-examples) repository. Example scripts should be added to that repository then the submodule pointer updated here.

## Markdown Linting

Basic markdown linting has been enabled to catch common spacing. Currently manually linting via the VS Code extension. Linting should be setup via ESlint and run on build.

-   [VS Code Plugin](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint)
-   [Rules](https://github.com/DavidAnson/markdownlint/tree/v0.34.0/doc)

### Broken links

To catch broken links between pages please enable Markdown Validation in your VS code setup.

-   navigate to preferences: Open Settings (UI)
-   search for `markdown validate`
-   Markdown > Validate: Enabled (the default is Disabled)
