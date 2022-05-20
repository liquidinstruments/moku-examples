
/*
    * Simple Java example on how to use the Moku REST API fir Liquid Instrument Moku Devices.
    * IMPORTANT: Deploy the Oscilloscope through the Desktop or iPad apps before
    *        running this script in order to transfer the instrument data. See
    *        https://apis.liquidinstruments.com/starting-curl.html#first-steps
    * NOTE: This example uses cliftonlabs json-simple library to serialize/de-serialize request 
    *      and response. For more information, please visit https://cliftonlabs.github.io/json-simple/
    * NOTE: This example demonstrates how to deploy and interact with an Oscilloscope. 
    *       Details on list of methods and associated request schemas can be found at 
    *       https://apis.liquidinstruments.com/reference/oscilloscope/ 
*/
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.util.Map;
import static java.util.Map.entry;
import java.net.http.HttpRequest.BodyPublishers;
import com.github.cliftonlabs.json_simple.JsonObject;
import com.github.cliftonlabs.json_simple.Jsoner;

public class MokuDemo {

    static String ipAddress = "192.168.73.1";

    private static JsonObject executeAPICall(HttpRequest request, HttpClient client) throws Exception {
        HttpResponse<String> getDataResponse = client.send(request, BodyHandlers.ofString());
        if (getDataResponse.statusCode() != 200) {
            String errorText = String.format("Error occurred. Response code: %d", getDataResponse.statusCode());
            throw new Exception(errorText);
        }
        JsonObject data = (JsonObject) Jsoner.deserialize(getDataResponse.body());
        if (!(Boolean) data.get("success")) {
            System.out.println(data.get("messages"));
        }
        Object returnData = data.get("data");
        if(returnData.getClass().getName() == "java.lang.String"){
            return (JsonObject) Jsoner.deserialize((String)returnData);
        }
        return (JsonObject) returnData;
    }

    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newHttpClient();

        /*
         * The first thing we need to do is to take ownership of the Moku.
         * We do that by sending a `POST` to the `claim_ownership` resource and it
         * gives us back a key that we can include in all future requests to prove we
         * are the owner.
         */
        String clientKeyURI = String.format("http://%s/api/moku/claim_ownership", ipAddress);
        HttpRequest clientKeyRequest = HttpRequest.newBuilder()
                .uri(URI.create(clientKeyURI))
                .headers("Content-Type", "application/json")
                .POST(BodyPublishers.noBody())
                .build();
        HttpResponse<String> clientKeyResponse = client.send(clientKeyRequest, BodyHandlers.ofString());
        String clientKey = clientKeyResponse.headers().firstValue("Moku-Client-Key").orElseThrow();

        // Get the Moku's name
        String mokuNameURI = String.format("http://%s/api/moku/name", ipAddress);
        HttpRequest nameRequest = HttpRequest.newBuilder()
                .uri(URI.create(mokuNameURI))
                .headers("Content-Type", "application/json")
                .headers("Moku-Client-Key", clientKey)
                .GET()
                .build();
        HttpResponse<String> nameResponse = client.send(nameRequest, BodyHandlers.ofString());
        System.out.println(nameResponse.body());

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

        JsonObject obj = new JsonObject().putAllChain(Map.ofEntries(
                entry("channel", 1),
                entry("range", "10Vpp"),
                entry("coupling", "AC"),
                entry("impedance", "1MOhm")));

        String frontendURI = String.format("http://%s/api/oscilloscope/set_frontend", ipAddress);
        HttpRequest frontendRequest = HttpRequest.newBuilder()
                .uri(URI.create(frontendURI))
                .headers("Content-Type", "application/json")
                .headers("Moku-Client-Key", clientKey)
                .POST(BodyPublishers.ofString(obj.toJson()))
                .build();
        JsonObject frontendResponse = executeAPICall(frontendRequest, client);
        // Print the impedance deserialized from response object
        System.out.println(frontendResponse.get("impedance"));
        
        /*
        * Now that instrument is deployed, we can read out a frame of data using the
        * get_data method which returns time series data for channels 1 & 2.
        */
        JsonObject getDataRequestBody = new JsonObject().putAllChain(
                Map.ofEntries(entry("wait_reacquire", true)));
        String getDataURI = String.format("http://%s/api/oscilloscope/get_data", ipAddress);
        HttpRequest getDataRequest = HttpRequest.newBuilder()
                .uri(URI.create(getDataURI))
                .headers("Content-Type", "application/json")
                .headers("Moku-Client-Key", clientKey)
                .POST(BodyPublishers.ofString(getDataRequestBody.toJson()))
                .build();
        JsonObject getDataResponse = executeAPICall(getDataRequest, client);
        System.out.println(getDataResponse.get("time"));
    }

}