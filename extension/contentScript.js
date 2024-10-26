// contentScript.js

// Load ONNX Runtime Web
const script = document.createElement('script');
script.src = chrome.runtime.getURL('ort.min.js');
document.head.appendChild(script);

script.onload = () => {
  main();
};

async function main() {
  // Load the model
  const session = await ort.InferenceSession.create(
    chrome.runtime.getURL('fake_review_model.onnx')
  );

  // Analyze reviews on the page
  analyzeReviews(session);
}

async function analyzeReviews(session) {
  // Select review elements from the page (you may need to adjust the selector to match the site's HTML)
  const reviews = document.querySelectorAll('.review-text'); // Update selector as needed

  for (const review of reviews) {
    try {
      const text = review.innerText;
      const inputTensor = await preprocessText(text);

      // Run the model
      const feeds = { input: inputTensor };
      const results = await session.run(feeds);
      const output = results.output.data;

      // Get predicted label
      const predictedLabel = output[0] > 0.5 ? 'OR' : 'CG'; // Assuming sigmoid output and threshold of 0.5

      // Overlay the result
      overlayResult(review, predictedLabel);
    } catch (error) {
      console.error('Error processing review:', error);
    }
  }
}

function overlayResult(reviewElement, label) {
  // Create an overlay element
  const overlay = document.createElement('div');
  overlay.textContent = label === 'OR' ? 'Original' : 'Fake';
  overlay.style.position = 'absolute';
  overlay.style.backgroundColor = label === 'OR' ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)';
  overlay.style.color = '#fff';
  overlay.style.padding = '5px';
  overlay.style.borderRadius = '5px';
  overlay.style.top = '0';
  overlay.style.right = '0';
  overlay.style.zIndex = '9999';

  // Ensure the parent element is positioned
  const parent = reviewElement.parentElement;
  const parentStyle = window.getComputedStyle(parent);
  if (parentStyle.position === 'static') {
    parent.style.position = 'relative';
  }

  // Append the overlay to the parent
  parent.appendChild(overlay);
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'analyze_reviews') {
      console.log('Content script received analyze_reviews message');

      // Assuming that each review has the class 'review'
      let reviews = document.querySelectorAll('.review');
      console.log('Number of reviews found:', reviews.length);

      if (reviews.length > 0) {
          reviews.forEach((review, index) => {
              // Placeholder logic for analysis
              const isFake = index % 2 === 0; // Example: mark every even review as fake

              // Create a label to display the result of analysis
              let label = document.createElement('div');
              label.style.fontWeight = 'bold';
              label.style.padding = '5px';
              label.style.marginTop = '5px';
              label.style.borderRadius = '3px';

              if (isFake) {
                  label.textContent = 'Fake Review';
                  label.style.backgroundColor = 'red';
                  label.style.color = 'white';
              } else {
                  label.textContent = 'Original Review';
                  label.style.backgroundColor = 'green';
                  label.style.color = 'white';
              }

              // Append the label to the review element
              review.appendChild(label);
              console.log(`Review ${index + 1} analyzed: ${isFake ? 'Fake' : 'Original'}`);
          });

          sendResponse({ success: true, analysis: 'Reviews found: ' + reviews.length });
      } else {
          console.log('No reviews found on the page.');
          sendResponse({ success: false, analysis: 'No reviews found.' });
      }
  }
});


