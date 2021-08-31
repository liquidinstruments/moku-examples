---
# prev: false
additional_doc: 
description: 
method: post
name: force connect
summary: What is force connect and when to use it
---

<headers/>

When you connect to your Moku to deploy an instrument, you can set force_connect to True or False. 

If force_connect is set to False, the API will first check if there is an existing connection with the Moku and establish a connection if there isn’t one, otherwise it will return an error. If force_connect is set to True, the API will terminate any existing connections and establish a new connection.

At the end of each session, you should always termination the session with [relinquish_ownership()](../moku/relinquish_ownership.md) so that other clients or a new API can connect to the Moku.

If you encounter an error stating a connection already exist, you can first try [relinquish_ownership()](../moku/relinquish_ownership.md) or set force_connect to True.