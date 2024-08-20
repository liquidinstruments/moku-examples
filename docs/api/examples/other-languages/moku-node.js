
/**
 * Let's Moku!
 *
 * We're going to poke and prod a little bit at the Moku's API server directly
 * using just node.js's `request` library. The below code assumes you are connected
 * to your Moku's WiFi access point. If this isn't true, replace the IP address
 * below with your Moku's address as read out of the Moku desktop app.
 *
 * This can be run using `node moku-node.js` with node v14+
 * First, run `npm install --save request`
**/

const request = require('request')

// Helper function for making HTTP requests asynchronously
// options is a dict containing url, method, headers dict (optional), body (optional)
async function makeRequest(options) {
    return new Promise((resolve, reject) => {
        request(options, (err, response, body) => {
            if (err) reject(err)
            else resolve({ body, response })
        })
    })
}

async function main() {
    /*
     * The first thing we need to do is to take ownership of the Moku.
     * We do that by sending a `POST` to the `claim_ownership` resource and it
     * gives us back a key that we can include in all future requests to prove we
     * are the owner.
     */
    const IP_ADDRESS = '192.168.73.1'
    let response = await makeRequest({
        url: `http://${IP_ADDRESS}/api/moku/claim_ownership`,
        method: 'POST',
    })
    console.log(JSON.parse(response.body))
    const clientKey = response.response.headers['moku-client-key']

    // Get the Moku's name
    response = await makeRequest({
        url: `http://${IP_ADDRESS}/api/moku/name`, method: 'GET',
        headers: { 'Moku-Client-Key': clientKey }
    })
    console.log(JSON.parse(response.body))

    /*
     * As you can see, all responses from the Moku are formatted as a JSON dictionary
     * with four properties. `success` can be true or false depending whether what
     * you asked for was valid. It it's true, `data` contains the value(s) you asked for.
     * If it's false then `code` and `messages` tell you why.
     *
     * The first time you access an instrument's resource, that instrument is deployed.
     * Here we set the Oscilloscope frontend, implicitly deploying it first.
     */

    response = await makeRequest({
        url: `http://${IP_ADDRESS}/api/oscilloscope/set_frontend`,
        method: 'POST',
        body: JSON.stringify({ channel: 1, range: '10Vpp', coupling: 'AC', impedance: '1MOhm' }),
        headers: { 'Moku-Client-Key': clientKey }
    })
    console.log(JSON.parse(response.body))

    response = await makeRequest({
        url: `http://${IP_ADDRESS}/api/oscilloscope/get_data`,
        method: 'POST',
        body: JSON.stringify({ wait_reacquire: false }),
        headers: { 'Moku-Client-Key': clientKey }
    })
    // This prints out a frame dictionary like so:
    // { time: [-0.005, -0.004, -0.003, ...], ch1: [0.0, 0.0, 0.0, ...], ... }
    console.log(JSON.parse(response.body).data)
}

main()
