use std::error::Error;

/**
 * Let's Moku!
 *
 * We're going to poke and prod a little bit at the Moku's API server directly
 * using just the `reqwest` and `json` Rust crates. The below code assumes you are connected
 * to your Moku's WiFi access point. If this isn't true, replace the IP address
 * below with your Moku's address as read out of the Moku desktop app.
 *
 * We use the `reqwest` crate to do blocking HTTP requests.
 * You'll need to add `reqwest` and `json` to your `Cargo.toml` file as follows:
 *   [dependencies]
 *   reqwest = { version = "0.11", features = ["blocking"] }
 *   json = { version = "0.12.4" }
 *
 * If you're starting a Rust project from scratch, you should:
 * 1. Run `cargo new mokurs` and `cd mokurs`
 * 2. Replace `src/main.rs` with the contents of this file
 * 3. Add the dependencies section above to `Cargo.toml`
 * 4. Run `cargo run`
**/

fn main() -> Result<(), Box<dyn Error>> {
    /*
     * The first thing we need to do is to take ownership of the Moku.
     * We do that by sending a `POST` to the `claim_ownership` resource and it
     * gives us back a key that we can include in all future requests to prove we
     * are the owner.
     */
    const IP_ADDRESS: &str = "192.168.73.1";
    let client = reqwest::blocking::Client::new();

    // We must supply an empty body, otherwise the request will be malformed
    let client_key: String = {
        let resp = client.post(format!("http://{IP_ADDRESS}/api/moku/claim_ownership"))
            .body("").send()?;
        let client_key: String = String::from(resp.headers().get("Moku-Client-Key").unwrap().to_str()?);
        println!("{}", json::parse(&resp.text()?[..])?);  // moves resp
        client_key
    };

    // Get the Moku's name
    {
        let resp = client.get(format!("http://{IP_ADDRESS}/api/moku/name"))
            .header("Moku-Client-Key", client_key.clone()).send()?;
        println!("{}", json::parse(&resp.text()?[..])?);
    }

    /*
     * As you can see, all responses from the Moku are formatted as a JSON dictionary
     * with four properties. `success` can be true or false depending whether what
     * you asked for was valid. It it's true, `data` contains the value(s) you asked for.
     * If it's false then `code` and `messages` tell you why.
     *
     * The first time you access an instrument's resource, that instrument is deployed.
     * Here we set the Oscilloscope frontend, implicitly deploying it first.
     */
    {
        let options = json::object!{
            channel: 1,
            range: "10Vpp",
            coupling: "AC",
            impedance: "1MOhm"
        };
        let resp = client.post(format!("http://{IP_ADDRESS}/api/oscilloscope/set_frontend"))
            .header("Moku-Client-Key", client_key.clone())
            .body(options.dump())
            .send()?;
        println!("{}", json::parse(&resp.text()?[..])?);
    }
    // Get a frame of data
    {
        let resp = client.post(format!("http://{IP_ADDRESS}/api/oscilloscope/get_data"))
            .header("Moku-Client-Key", client_key.clone())
            .body("{\"wait_reacquire\": false}")
            .send()?;
        let result: json::JsonValue = json::parse(&resp.text()?[..])?;
        // We print out result['data']['ch1'], which is an array of floats
        if let json::JsonValue::Object(res_obj) = result {
            if let json::JsonValue::Object(data_obj) = res_obj.get("data").unwrap() {
                println!("{}", data_obj.get("ch1").unwrap());
            }
        }
    }

    Ok(())
}
