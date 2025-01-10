# Getting Started with cURL, REST

The API on the Moku is, at its core, a simple RESTful API. The API packages provided by Liquid Instruments provide friendly wrappers around this interface for common languages, but it's quite possible to access the REST interface directly. This allows you to easily interact with your Moku from a terminal or, indeed, any programming language at all that supports REST (or even just HTTP).

This page describes how to use the API from a Command Line using cURL; in particular, it is assumed you're using a `bash`-like terminal as Windows CMD or PS may require different character escaping.

:::warning Preliminary
The language wrappers provide convenience services that are not provided by the raw RESTful API described here. Please make sure you read the following steps carefully and note that they are subject to change.
:::

## First Steps

The Moku requires a number of data files for correct operation. The `Moku:` applications and the language wrappers bundle this data but it is not currently available separately. As such, you _must_ have opened an instrument from an application at least once since the Moku was powered on before it's available from this API.

:::warning Data Files
This is a common source of errors. If you see responses with the code `NO_BIT_STREAM` then this is the cause. Open the target instrument from a `Moku:` application and try again (you may close the application as soon as the instrument control screen appears).
:::

## IP addresses and URLs

Before you start, you must find the IP address of your Moku and confirm that you can connect to it. See [this page](./ip-address.md) for more details.

URLs in the REST API take the form

```
http://<ip>/api/<instrument>/<action>
```

For example `http://10.1.1.1/api/awg/set_defaults`. Interaction is generally by `POST`ing a JSON structure to that URL.

## Getting a Client Key

The Moku has a concept of "ownership" to prevent people unintentionally changing settings on a Moku that someone else is using. In the REST API, this means that a user must first get a Client Key, then supply that key with each future request.

To get a Client Key, `POST` an empty JSON object to the `moku/claim_ownership` endpoint. The Client Key will be returned in the Response Header, use `--include` to print those headers.

```bash
$: curl --include \
        --data '{}'\
        -H 'Content-Type: application/json'\
        http://<ip>/api/moku/claim_ownership

HTTP/1.1 200 OK
...
Moku-Client-Key: 17cfd311ecb

{"success":true,"data":null,"code":null,"messages":null}
```

Record the `Moku-Client-Key` header and include it in all future requests.

## Performing Operations

See the [API Reference](../reference/) pages for details on the endpoints. The first time you access any endpoint for a particular instrument, a "deploy" will be triggered, configuring the Moku to use that instrument. This may take several seconds.

All actions are `POST`s of a JSON object. The JSON object contains the parameters for the action, if any; an empty object must be passed if the action doesn't take any parameters. For example

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1}' \
        http://10.1.111.193/api/awg/disable_modulation
```

## Releasing the Moku (optional)

When you've finished, you _should_ release your ownership of the Moku. If you do not do this, then the device will continue to appear "In Use" in the apps and users will be prompted to confirm that they really want to connect.

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/moku/relinquish_ownership
```

## Decoding Output with `jq`

The return values from the commands will be a JSON structure. At the top level, it _always_ as the following elements:

-   _success_ [true|false]: Whether or not the command has been accepted by the Moku
-   _code_ [string]: If `success` is `false`, the error code is presented in this field
-   _messages_ [string]: If `success` is `false`, a human-readable error message, or series of error messages, is present here
-   _data_ [object|string]: If `success` is `true`, the return value is present here

The command line JSON Query command `jq` can be used to isolate these elements and those inside.

```bash
# Extract the magnitude data from a Frequency Response Analyzer trace
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/fra/get_data | \
        jq ".data.ch1.magnitude"
[ ... <list of floats> ]
```

```bash
# Isolate the error message from passing an invalid channel
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 6}'\
        http://<ip>/api/fra/disable_output | \
        jq ".messages[0]"
"Invalid channel. Should be in between 1 and 2"
```

## Next Steps

Check out the cURL examples for each endpoint in the [API Reference](../reference/).

## Troubleshooting

#### IPv6 (including USB) Connection Issues

There are some environmental limitations when using IPv6, including using the Moku USB interface. See [this section](./ip-address.md#ipv6) for more information.
