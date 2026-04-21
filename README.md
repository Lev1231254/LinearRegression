<h1>Linear Regression from Scratch (California Housing)</h1>

<p>
This is a small training project where I built a linear regression model from scratch
using the California Housing dataset.
</p>

<h2>What’s inside</h2>

<ul>
  <li><b>Gradient Descent</b> – used to train the model</li>
  <li><b>L1 Regularization</b> – pushes some weights toward zero</li>
  <li><b>L2 Regularization</b> – shrinks weights to help avoid overfitting</li>
  <li><b>Feature Scaling</b> – centers features so their median is zero</li>
  <li><b>Validation</b> – checks performance on separate data</li>
</ul>

<h2>Analysis</h2>

<p>
The goal is to predict the average number of rooms using these features:
</p>

<p>
MedInc, HouseAge, AveBedrms, Population, AveOccup, Latitude, Longitude
</p>

<h3>No Regularization vs L1 + L2</h3>

<img width="1536" height="850" alt="normal + L1L2"
src="https://github.com/user-attachments/assets/f0647f74-6b34-49fc-a493-eed3a2e1e7fd" />

<p>
Without regularization, the model performs slightly better on the test set,
but it takes about twice as long to train.
</p>

<p>With L1 + L2 regularization, you can see that:</p>

<ul>
  <li>Less important features get smaller weights</li>
  <li>Some weights go close to zero (mostly because of L1)</li>
</ul>

<h3>Feature Importance</h3>

<p><b>Strongest correlations:</b></p>
<ul>
  <li>Average Bedrooms</li>
  <li>Median Income</li>
</ul>

<p><b>Weakest correlations:</b></p>
<ul>
  <li>Population</li>
  <li>Longitude</li>
  <li>House Age</li>
</ul>

<p><b>Interesting:</b></p>
<ul>
  <li>Latitude has a noticeable effect on the target</li>
  <li>Average Occupancy doesn’t really matter much</li>
</ul>

<p>
One possible reason is that some areas in California are simply wealthier,
so houses there tend to be bigger.
</p>

<h3>L1 vs L2 Regularization</h3>

<img width="1536" height="850" alt="L1 + L2"
src="https://github.com/user-attachments/assets/d7dd5419-dc6a-424b-8184-cf003720a039" />

<ul>
  <li>L2 works better overall</li>
  <li>L1 pushes some weights almost to zero (basically removing some features)</li>
  <li>L2 spreads the weights more smoothly:
    <ul>
      <li>Important features get slightly smaller weights</li>
      <li>Less important ones still contribute a bit</li>
    </ul>
  </li>
</ul>

<h2>Conclusion</h2>

<p>
All models perform pretty similarly, but L2 gives the best results overall.
It keeps things balanced without completely removing features.
</p>
