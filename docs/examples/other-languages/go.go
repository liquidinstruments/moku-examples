// Simple example of interacting with a Moku REST API from Go.
// This prioritises simplicity over, e.g. type safety so request bodies are formed
// as string and returns are unmarshalled to maps rather than structs.

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
)

// These fields are case-insensitively matched against the JSON field names and therefore
// capitalized according to Go convention rather than exactly matching the JSON.
type moku_response struct {
	Success  bool
	Data     interface{}
	Messages []string
	Code     string
}


func do_rest_request(base_url, client_key, endpoint string, body []byte) (interface{}, error) {
	client := &http.Client{}

	req, err := http.NewRequest("POST", base_url+endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Add("content-type", "application/json")

	if client_key != "" {
		req.Header.Add("Moku-Client-Key", client_key)
	}

	resp, err := client.Do(req)

	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, readErr := ioutil.ReadAll(resp.Body)
	if readErr != nil {
		return nil, readErr
	}

	response := moku_response{}
	jsonErr := json.Unmarshal(respBody, &response)

	if jsonErr != nil {
		return nil, jsonErr
	}

	if response.Success == false {
		return nil, errors.New(response.Messages[0])
	}

	return response.Data, nil
}


func take_ownership(base_url string) (string, error) {
	data, err := do_rest_request(base_url, "", "moku/claim_ownership", nil)

	if err != nil {
		return "", err
	}

	return data.(string), nil
}


func relinquish_ownership(base_url, client_key string) error {
	_, err := do_rest_request(base_url, client_key, "moku/relinquish_ownership", nil)
	return err
}


func main() {
	// Change your IP address here
	base_url := "http://192.168.73.1/api/"

	// The data returned from the take ownership request is exactly the client key (as well as the key
	// being present in a header)
	client_key, err := take_ownership(base_url)
	if err != nil {
		log.Fatal(err)
	}
	defer relinquish_ownership(base_url, client_key)

	// Now that we have ownership, and a client key to prove it, we can issue requests to any other endpoint we like.
	// Here the JSON body is formed directly as a byte array, you can also use the JSON library to marshall a struct
	// with the parameters if you prefer to keep strong typing on the requests.
	_, err = do_rest_request(base_url, client_key, "oscilloscope/set_frontend",
		[]byte(`{"channel": 1, "impedance": "1MOhm", "coupling": "AC", "range": "10Vpp"}`))


	data, err := do_rest_request(base_url, client_key, "oscilloscope/get_data", []byte(`{"wait_reacquire": false}`))

	// At this point, the actual request-specific response data (right now, frame data) is just in a map.
	// If you want to maintain type safety, you can:
	// 1. Open-code more of the do_rest_request so you can pass a response-specific struct to the unmarshalling
	// 2. Re-structure it with e.g https://github.com/mitchellh/mapstructure
	// 3. Make do_rest_response generic on the return structure (requires Go 1.18+, still in beta at the time of writing)
	fmt.Println(data.(map[string]interface{})["ch1"])
}
