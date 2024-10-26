document.addEventListener('DOMContentLoaded', () => {
    console.log('Popup DOM fully loaded and parsed');
    
    const analyzeButton = document.getElementById('analyze-button');
    const resultDiv = document.getElementById('result');

    if (analyzeButton && resultDiv) {
        analyzeButton.addEventListener('click', () => {
            console.log('Analyze button clicked');

            // Query the active tab to get its ID
            chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                if (tabs.length > 0) {
                    const tabId = tabs[0].id;

                    // Try injecting the content script
                    chrome.scripting.executeScript({
                        target: { tabId: tabId },
                        files: ['contentScript.js']
                    }, (injectionResults) => {
                        if (chrome.runtime.lastError) {
                            console.error('Error injecting content script:', chrome.runtime.lastError.message);
                            resultDiv.textContent = 'Failed to inject content script: ' + chrome.runtime.lastError.message;
                            return;
                        }

                        // After successful injection, send a message to the content script
                        console.log('Sending analyze_reviews message to content script');
                        chrome.tabs.sendMessage(tabId, { type: 'analyze_reviews' }, (response) => {
                            if (chrome.runtime.lastError) {
                                // This will catch cases where the content script isn't available or the message fails
                                console.error('Error sending message:', chrome.runtime.lastError.message);
                                resultDiv.textContent = 'Failed to analyze reviews: ' + chrome.runtime.lastError.message;
                            } else if (response && response.success) {
                                resultDiv.textContent = 'Analysis complete: ' + response.analysis;
                            } else {
                                resultDiv.textContent = 'Failed to analyze reviews.';
                            }
                        });
                    });
                } else {
                    console.error('No active tab found');
                    resultDiv.textContent = 'No active tab found.';
                }
            });
        });
    } else {
        console.error('Analyze button or result div not found');
    }
});
