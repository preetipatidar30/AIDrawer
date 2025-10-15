require("dotenv").config();
const express = require("express");
const app = express();
const port = 4000;
const fetch = require("node-fetch");
const bodyParser = require("body-parser");
const cors = require("cors");

app.use(bodyParser.json());
app.use(cors());

// Example: Update relevant HTML elements 
function updateUI(data) {
    const domainElement = document.getElementById('domain');
    const outputElement = document.getElementById('output');
    const errorElement = document.getElementById('error');
    const timeElement = document.getElementById('time');

    if (domainElement) domainElement.textContent = data.domain;
    if (outputElement) outputElement.textContent = data.output;
    if (errorElement) errorElement.textContent = data.error;
    if (timeElement) timeElement.textContent = `Time Taken: ${data.time_taken} seconds`;
}

async function sendMessage(message) { 
    try {
        const response = await fetch("http://127.0.0.1:5000", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: message }) 
        });

        if (response.ok) {
            const data = await response.json();
            updateUI(data); // Update UI based on JSON response
        } else {
            const errorData = await response.json(); 
            console.error("Error from server:", errorData);
            // Handle error display in the UI
        }
    } catch (error) {
        console.error("Error calling server:", error);
        // Handle general errors
    }
}

// ... Input field and button logic to call sendMessage on user action

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`);
});
