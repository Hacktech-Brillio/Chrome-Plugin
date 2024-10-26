// Listener for when the user clicks on the extension's action icon (popup)
chrome.action.onClicked.addListener((tab) => {
    // For example, when the extension icon is clicked, it can inject a content script into the current page
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['contentScript.js']
    });
});

// Message listener to handle messages from other parts of the extension
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'analyze_reviews') {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, message, (response) => {
                sendResponse(response);
            });
        });
        return true;
    }
});

// Storage example: Handling settings or data for the extension
chrome.runtime.onInstalled.addListener(() => {
    // Set default values in Chrome storage when the extension is first installed
    chrome.storage.sync.set({ modelLoaded: false }, () => {
        console.log('Default modelLoaded value set to false.');
    });
});
