/*
 * Simple C++ example on how to use the Moku REST API for Liquid Instruments
 * Moku Devices.
 * IMPORTANT: Deploy the Oscilloscope through the Desktop or iPad apps before
 *        running this script in order to transfer the instrument data. See
 *        https://apis.liquidinstruments.com/starting-curl.html#first-steps
 * NOTE: This example demonstrates how to deploy and interact with an
 *       Oscilloscope. Details on list of methods and associated request
 *       schemas can be found at
 *       https://apis.liquidinstruments.com/reference/oscilloscope/
 * NOTE: This example uses nlohmann/json(https://json.nlohmann.me/) which is a header only library
 *       to serialize and deserialize JSON body.
 */
#include <iostream>
#include <curl/curl.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

std::string ipAddress = "192.168.73.1";
std::string clientKey = "";

std::string buildURL(const std::string &endpoint)
{
    std::ostringstream urlStream;
    urlStream << "http://" << ipAddress << "/api/" << endpoint;
    return urlStream.str();
}

json executeHttpRequest(CURL *curl, const std::string &url, const std::string &response)
{
    CURLcode resCode = curl_easy_perform(curl);

    if (resCode != CURLE_OK)
    {
        fprintf(stderr, "Error occurred while performing %s. Error: %s", url.c_str(), curl_easy_strerror(resCode));
        std::exit(-1);
    }
    else
    {
        long httpCode;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        if (httpCode != 200)
        {
            fprintf(stderr, "Error occurred while performing %s. HTTP Code: %ld", url.c_str(), httpCode);
            std::exit(-1);
        }
        json apiResponse = json::parse(response);
        if (!apiResponse["success"])
        {
            std::ostringstream errorStream;
            errorStream << apiResponse["code"] << ": " << apiResponse["messages"].dump();
            fprintf(stderr, "%s", errorStream.str().c_str());
            std::exit(-1);
        }
        return apiResponse["data"];
    }
}

size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *response)
{
    size_t totalSize = size * nmemb;
    response->append(static_cast<char *>(contents), totalSize);
    return totalSize;
}

json httpGet(CURL *curl, const std::string &url)
{
    CURLcode resCode;
    std::string response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    return executeHttpRequest(curl, url, response);
}

json httpPost(CURL *curl, const std::string &url, json requestBody)
{
    CURLcode resCode;
    std::string response;

    auto postData = requestBody.dump();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, postData.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    return executeHttpRequest(curl, url, response);
}

int main()
{
    CURL *curl;
    CURLcode res;

    curl = curl_easy_init();

    if (curl)
    {
        {
            /*
             * The first thing we need to do is to take ownership of the Moku.
             * We do that by sending a `POST` to the `claim_ownership` resource
             * and it gives us back a key that we can include in all future
             * requests to prove we are the owner.
             */
            json claimReq;
            claimReq["force_connect"] = true;
            clientKey = httpPost(curl, buildURL("moku/claim_ownership"), claimReq);

            /*
             * Now that we obtained the clientKey, we include it as
             * part of header for every subsequent request proving
             * the ownership.
             * Let's do that by adding it to the headers of
             * the curl object
             */
            struct curl_slist *headers = NULL;
            std::ostringstream clientKeyHeader;
            clientKeyHeader << "Moku-Client-Key: " << clientKey;

            headers = curl_slist_append(headers, clientKeyHeader.str().c_str());
            headers = curl_slist_append(headers, "Content-Type: application/json");

            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

            printf("Established connection, client key: %s\n", clientKey.c_str());
        }

        {
            // Get the Moku's name
            auto result = httpGet(curl, buildURL("moku/name"));
            std::cout << result << std::endl;
        }

        {
            // Get the Moku's serial
            auto result = httpGet(curl, buildURL("moku/serial_number"));
            std::cout << result << std::endl;
        }

        {
            json frontendRequest = {{"channel", 1}, {"coupling", "DC"}, {"impedance", "1MOhm"}, {"range", "10Vpp"}};

            auto result = httpPost(curl, buildURL("slot1/oscilloscope/set_frontend"), frontendRequest);
            std::cout << result << std::endl;
        }

        {
            // Get data
            json getDataRequest = {{"wait_reacquire", true}};

            auto result = httpPost(curl, buildURL("slot1/oscilloscope/get_data"), getDataRequest);
            std::cout << result << std::endl;
        }

        // Clean up
        curl_easy_cleanup(curl);
    }

    // Clean up global cURL resources
    curl_global_cleanup();

    return 0;
}
