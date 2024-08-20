# Simple example of using the Moku REST API for Liquid Instruments Moku devices.
# IMPORTANT: Deploy the Oscilloscope through the Desktop or iPad apps before
#            running this script in order to transfer the instrument data. See
#            https://apis.liquidinstruments.com/starting-curl.html#first-steps


use warnings;
use strict;

use Data::Dumper;

use JSON;
use REST::Client;

# Create a REST Client pointed at our Moku's IP address.
# All Moku REST endpoints expect (and return) JSON bodies, so we attach that header
# permanently to the client (rather than passing per-request).
my $client = REST::Client->new();
$client->setHost('http://192.168.73.1');
$client->addHeader('content-type', 'application/json');

# The first step with the Moku REST API is to claim ownership. A POST to this endpoint
# returns a client key which we also permanently attach to the client so it gets presented
# on each later call. You can also pass the request body '{"force_connect": true}' if you
# want to take over ownership regardless of the current ownership status.
my $key = $client->POST('/api/moku/claim_ownership')->responseHeader('Moku-Client-Key');
$client->addHeader('Moku-Client-Key', $key);

# Do a few operations targeting the Oscilloscope instrument. The bodies can be marshalled/
# /unmarshalled to/from JSON using the JSON library, or can be presented statically as a string
# if you prefer (e.g. if it's all constants anyway).
$client->POST('/api/oscilloscope/set_frontend',
              '{"channel": 1, "impedance": "1MOhm", "range": "10Vpp", "coupling": "DC"}'
             );

my $responseFrame = from_json $client->POST('/api/oscilloscope/get_data',
                                            '{"wait_reacquire": false}'
                                           )->responseContent();
my %dataFrame = %{$responseFrame->{data}};

# Just print the keys of the returned data frame, in a real program you'd do something with the data!
print Dumper(keys %dataFrame);

# Don't forget to relinquish ownership at the end. If you miss this line, e.g. because a line
# above croaks, then you (and anyone else) will need to force_connect next time around.
$client->POST('/api/moku/relinquish_ownership');
