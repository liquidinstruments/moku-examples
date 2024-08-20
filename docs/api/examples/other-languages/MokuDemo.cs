/*
    * Simple C# example on how to use the Moku REST API for Liquid Instrument 
    * Moku Devices.
    * IMPORTANT: Deploy the Oscilloscope through the Desktop or iPad apps before
    *        running this script in order to transfer the instrument data. See
    *        https://apis.liquidinstruments.com/starting-curl.html#first-steps
    * NOTE: This example demonstrates how to deploy and interact with an 
    *       Oscilloscope. Details on list of methods and associated request 
    *       schemas can be found at 
    *       https://apis.liquidinstruments.com/reference/oscilloscope/ 
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading.Tasks;

class MokuDemo
{
    const string IP_ADDRESS = "192.168.73.1";
    static Uri baseUri = new Uri(String.Format("http://{0}/api/", IP_ADDRESS));

    static async Task Main(string[] args)
    {
        /* Create a HttpClient object, we will be using the 
         * same object to send every request */

        HttpClient client = new HttpClient();
        client.BaseAddress = baseUri;
        client.DefaultRequestHeaders.Accept.Add(
            new MediaTypeWithQualityHeaderValue("application/json"));

        /*
          * The first thing we need to do is to take ownership of the Moku.
          * We do that by sending a `POST` to the `claim_ownership` resource 
          * and it gives us back a key that we can include in all future 
          * requests to prove we are the owner.
        */

        HttpResponseMessage clientKeyResponse = await client.PostAsync(
            "moku/claim_ownership", null);
        if (clientKeyResponse.IsSuccessStatusCode)
        {
            IEnumerable<string> headerValues = new string[] { };
            clientKeyResponse.Headers.TryGetValues("Moku-Client-Key", out headerValues);
            /* 
             * Now that we obtained the clientKey, we include it as
             * part of header for every subsequent request proving
             * the ownership.
             * Let's do that by adding it to the DefaultRequestHeaders of
             * the HttpClient object
             */
            client.DefaultRequestHeaders.Add("Moku-Client-Key", headerValues.First());
        }
        else
        {
            Console.WriteLine("Unexpected error: HTTP Response Code {0}", clientKeyResponse.StatusCode);
        }

        // Get the Moku's name

        HttpResponseMessage nameResponse = await client.GetAsync("moku/name");
        if (nameResponse.IsSuccessStatusCode)
        {
            string responseContent = nameResponse.Content.ReadAsStringAsync().Result;
            var nameObject = JsonDocument.Parse(responseContent);
            Console.WriteLine("Moku name is '{0}'", nameObject.RootElement.GetProperty("data"));
        }
        else
        {
            Console.WriteLine("Unexpected error: HTTP Response Code {0}", nameResponse.StatusCode);
        }

        /*
         * As you can see, all responses from the Moku are formatted as a JSON
         * dictionary
         * with four properties. `success` can be true or false depending whether what
         * you asked for was valid. It it's true, `data` contains the value(s) you asked
         * for.
         * If it's false then `code` and `messages` tell you why.
         *
         * The first time you access an instrument's resource, that instrument is
         * deployed.
         * Here we set the Oscilloscope frontend, implicitly deploying it first.
         */
        var frontendRequest = new
        {
            channel = 1,
            range = "10Vpp",
            coupling = "AC",
            impedance = "1MOhm"
        };
        StringContent frontendRequestContent = new StringContent(JsonSerializer.Serialize(frontendRequest));
        HttpResponseMessage frontendResponse = await client.PostAsync("oscilloscope/set_frontend", frontendRequestContent);
        if (frontendResponse.IsSuccessStatusCode)
        {
            string responseContent = frontendResponse.Content.ReadAsStringAsync().Result;
            var frontendObject = JsonDocument.Parse(responseContent).RootElement.GetProperty("data");
            var frontendParams = JsonSerializer.Deserialize<JsonDocument>(frontendObject.ToString());
            Console.WriteLine("Input range configured to : {0}", frontendParams.RootElement.GetProperty("range"));
        }
        else
        {
            Console.WriteLine("Unexpected error: HTTP Response Code {0}", frontendResponse.StatusCode);
        }

        //Get Data
        StringContent getDataContent = new StringContent(JsonSerializer.Serialize(new { wait_reacquire = false }));
        HttpResponseMessage getDataResponse = await client.PostAsync("oscilloscope/get_data", getDataContent);
        if (getDataResponse.IsSuccessStatusCode)
        {
            string responseContent = getDataResponse.Content.ReadAsStringAsync().Result;
            var getDataObject = JsonDocument.Parse(responseContent).RootElement.GetProperty("data");

            Console.WriteLine("Channel1 data: {0}", getDataObject.GetProperty("ch1"));
        }
        else
        {
            Console.WriteLine("Unexpected error: HTTP Response Code {0}", getDataResponse.StatusCode);
        }

    }
};

