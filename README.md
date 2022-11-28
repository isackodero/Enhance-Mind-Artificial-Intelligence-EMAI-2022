# Enhance-Mind-Artificial-Intelligence-EMAI-2022
<h1>DEEP LEARNING FUNDAMENTALS</h1> <br><br>
<h4>Isack Odero<h4/>
<img src='img/zoom.jpg'><br><h3>What is Deep Learning</h3><br><img src='img/Screenshot from 2022-11-15 21-56-18.png'><br><h3>Why Deep Learning and Why Now ?</h3><br><h4>Why Deep Learning</h4><br><p>Hand engineer features are<li>Time consuming</li><li>Brittle</li><li>Not scalable in practice</li></p><br><img src='img/Screenshot from 2022-11-15 22-08-35.png'><br><div><h4>Why Now ? </h4><br><img src='img/Screenshot from 2022-11-15 22-30-22.png'><br><p>Neural Networks dates back a decades, so why now?</p><br><p><lh>Big Data<lh><li>Large Datasets</li><li>Easier Collection</li><li>Storage</li> <br><img src='img/Screenshot from 2022-11-17 13-48-43.png'></p><br><p><lh>Hardware<lh><li>Graphics</li><li>Processing Units(GPU)</li><li>Massive Parallelizable</li> <br><img src='img/Screenshot from 2022-11-17 13-50-24.png'></p><br><p><lh>Software<lh><li>Improved Techniques</li><li>New Models</li><li>Toolboxes</li><br><img src='img/Screenshot from 2022-11-17 13-51-11.png'></p></div><br><h3 >The Perceptron</h3><br><h4>The structural building block of deep learning</h4></p><br><h5>The Perceptron: Forward Propagation</h5><br><img src='img/Screenshot from 2022-11-15 22-45-20.png'><br><img src='img/Screenshot from 2022-11-15 22-46-13.png'><br><img src='img/Screenshot from 2022-11-15 22-48-18.png'><br><img src='img/Screenshot from 2022-11-15 22-49-18.png'><br><img src='img/Screenshot from 2022-11-15 22-50-13.png'><br><h5>Activation Function</h5><br><img src='img/Screenshot from 2022-11-15 22-52-29.png'><br><h4>Common Activation Function</h4><br><img src='img/Screenshot from 2022-11-15 22-54-12.png'><br><h4>Importance of Activation Functions</h4><br><p>The purpose of activation function is to Intrduce <b>non-Linearities</b> into the network</p><br><img src='img/Screenshot from 2022-11-15 22-58-28.png'><br><img src='img/Screenshot from 2022-11-15 22-59-24.png'><br><img src='img/Screenshot from 2022-11-15 23-00-23.png'><br><h4>The Perceptron: Example</h4><br><img src='img/Screenshot from 2022-11-15 23-12-39.png'><br><img src='img/Screenshot from 2022-11-15 23-13-36.png'><br><img src='img/Screenshot from 2022-11-15 23-14-55.png'><br><img src='img/Screenshot from 2022-11-16 00-11-20.png'><br><img src='img/Screenshot from 2022-11-16 00-12-44.png'>
   <img src='img/Screenshot from 2022-11-15 23-27-08.png'>
   <h3>Building Neural Networks with Perceptron</h3>
   <h4>The Perceptron: Simplified</h4><br><img src='img/Screenshot from 2022-11-15 23-35-26.png'>
   <img src='img/Screenshot from 2022-11-15 23-36-37.png'>
   <img src='img/Screenshot from 2022-11-15 23-37-28.png'>
   
   
   ```python
   import torch
   import torch.nn as nn 
   ```
   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using {device} device")
   ```
  ```python
  Using cpu device
   
  ```
  ```python
  class Perceptron(nn.Module):\n",
        def __init__(self):
            super().__init__()
            self.flatten=nn.Flatten()
            self.layers=nn.Sequential(
                nn.Linear(2,1),
                nn.Sigmoid()
            )
    
        def forward(self, x):
            x=self.flatten(x)
            y_hat=self.layers(x)
            return y_hat
   ```
   
   ```python
   model=Perceptron().to(device)
   print(model
   ```
   
   ```python
   
   x=torch.tensor([[-1,2]], dtype=torch.float)
   
   y=model(x)
   print(y)
   
   ```
   
   <h3>Single Layer Neural Network</h3><br><img src='img/Screenshot from 2022-11-16 00-50-55.png'><br><img src='img/Screenshot from 2022-11-16 00-53-02.png'><br><img src='img/Screenshot from 2022-11-16 00-53-02.png'>
   <img src='img/Screenshot from 2022-11-16 00-54-38.png'><br><img src='img/Screenshot from 2022-11-16 00-55-48.png'><br><img src='img/Screenshot from 2022-11-16 00-56-24.png'>
   
   ```python
   
   class SingleLayerNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten=nn.Flatten()
            self.layers=nn.Sequential(
                nn.Linear(32*32,200),
                nn.ReLU(),
                nn.Linear(200, 2)
            )
           
        def forward(self, x):
            x=self.flatten(x)
            y_hat=self.layers(x)
            return y_hat        
   ```
   
   ```python
   
    model=SingleLayerNN()
    
    print(model)
   
   ```
   ```python
   
   image=torch.rand(1,32,32)
   
   y=model(image)
   print(y)
   
   ```
   
   <h3>Deep Neural Network</h3><br><img src='img/Screenshot from 2022-11-16 01-14-00.png'><br><img src='img/Screenshot from 2022-11-16 01-14-33.png'>
   
   ```python
   
   class DeepNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten=nn.Flatten()
            self.layers=nn.Sequential(
                nn.Linear(32*32,200),\
                nn.ReLU(),
                nn.Linear(200, 200),\
                nn.ReLU(),
                nn.Linear(200,100),
                nn.ReLU(),
               nn.Linear(100,10)
           )
           
        def forward(self, x):
            x=self.flatten(x)
            y_hat=self.layers(x)
            return y_hat
  ```
   ```python
   
   model=DeepNN()
   print(model)
   
   ```
   
   ```python
   y=model(image)
   
   print(y)
   ```
  
 <h4>Applying Neural Networks</h4><br><br><h5>Example Problem</h5><br><br><h5>Will You Pass if we decide to give an exam about the EMAI Conferrence presentations</h5><br><p>Let's start with a simple two feature model</p>
 <br><br><p>X_1= Number of lectures you attend<br> X_2 = Hours spent on the final project</p><br><img src='img/Screenshot from 2022-11-16 01-31-53.png'><br><img src='img/Screenshot from 2022-11-16 01-32-39.png'><br><img src='img/Screenshot from 2022-11-16 01-33-36.png'><br><img src='img/Screenshot from 2022-11-16 01-34-41.png'>
<h3>Quantifying Loss</h3><br><p>The LOSS of this will measures the cost incurrect predictions<p/><br><img src='img/Screenshot from 2022-11-16 01-38-12.png'>
   <h3>Cost Function</h3><br><p>The Cost Function measures the total loss over our entire datasets</p><br><img src='img/Screenshot from 2022-11-16 01-39-58.png'><br><img src='img/Screenshot from 2022-11-16 01-40-54.png'>
<p>Here we will look two types of Loss</p><br><li>Binary Cross Entropy Loss</li><li>Mean Squared Error Loss</li>
<h5>Binary Cross Entropy Loss</h5><br><p>This is used with the models that output a probability between 0 and 1</p><br><img src='img/Screenshot from 2022-11-16 01-48-41.png'><br><img src='img/Screenshot from 2022-11-16 01-49-29.png'>
   
   ```python
   
   Loss=nn.CrossEntropyLoss()
   
   ```
<h5>Mean Squared Error Loss</h5><br><p>This is used with the regression models that output continuous real number</p><br><img src='img/Screenshot from 2022-11-15 22-07-39.png'><br>
   
   ```python
   
   Loss = nn.MSELoss()
   
   ```
   <h3>Deep Neural Network</h3><br><h4>Loss Optimization</h4><br><p>We want to find the network weights that achieve the lowest loss</p><br><img src='img/Screenshot from 2022-11-15 22-18-41.png'><br><img src='img/Screenshot from 2022-11-15 22-19-37.png'>
   
   <img src='img/Screenshot from 2022-11-15 22-20-20.png'>
   <img src='img/Screenshot from 2022-11-15 22-22-13.png'>
   <img src='img/Screenshot from 2022-11-15 22-22-54.png'>
   <h5>If you will repeat the above process, that is what we call Gradient Descent</h5>"
  <img src='img/Screenshot from 2022-11-15 22-23-41.png'>
  <h3>Gradient Descent</h3>
   <lh><b>Algorithm</b></lh><br><li>Initialize weights randomly <img src='img/Screenshot from 2022-11-15 22-26-45.png'></li><li>Loop until converging</li><br><li>Compute Gradient <img src='img/Screenshot from 2022-11-15 22-31-09.png'></li><li>Update weights <img src='img/Screenshot from 2022-11-15 22-32-21.png'></li><li>Return weights</li><br><img src='img/Screenshot from 2022-11-17 13-14-22.png'>
   <p>This approach have one problem : the gradients can be very Computationally Intensive to compute</p>
   <h3>Gradient Descent</h3>
   <lh><b>Algorithm</b></lh><br><li>Initialize weights randomly <img src='img/Screenshot from 2022-11-15 22-26-45.png'></li><li>Loop until converging</li><br><li>Pick single data point <img src='img/Screenshot from 2022-11-17 13-19-51.png'> </li><br><li>Compute Gradient <img src='img/Screenshot from 2022-11-17 13-20-58.png'></li><li>Update weights <img src='img/Screenshot from 2022-11-15 22-32-21.png'></li><li>Return weights</li><br><img src='img/Screenshot from 2022-11-17 13-14-22.png'>
  <p>Now using <br><li>Compute Gradient <img src='img/Screenshot from 2022-11-17 13-20-58.png'></li> <br>it become easy to compute but the problem is it is produce very noisy (stochastic)</p><br><p>Now to solve that problem the Stochastic Gradient Descent is used</p>
   <h3>Stochastic Gradient Descent</h3>
    <lh><b>Algorithm</b></lh><br><li>Initialize weights randomly <img src='img/Screenshot from 2022-11-15 22-26-45.png'></li><li>Loop until converging</li><br><li>Pick batch of <i>B</i> data points </li><br><li>Compute Gradient <img src='img/Screenshot from 2022-11-17 13-31-49.png'></li><li>Update weights <img src='img/Screenshot from 2022-11-15 22-32-21.png'></li><li>Return weights</li><br><img src='img/Screenshot from 2022-11-17 13-14-22.png'><br><p><i>Now this become Fast to compute and a much better estimate of the true gradient</i></p>
    <h3>Mini-batches while training</h3><br><h4>More accurate estination of gradient</h4><br><li>Smoother convergen</li><br><li>Allows for larger learning rates</li>
    <br><h4>Mini-batches lead to fast trianing!</h4><br><li>can parallelize cotation  +   achieve significant speed increases on GPU's</li><h3>Computing Gradients: Backpropagation</h3>
    <img src='img/Screenshot from 2022-11-17 02-12-43.png'><br><p>How does a small change in one weight (eg.W_2) affect the final loss J(W)?</p><br><img src='img/Screenshot from 2022-11-17 02-15-05.png'><img src='img/Screenshot from 2022-11-17 02-17-10.png'><br><img src='img/Screenshot from 2022-11-17 02-18-22.png'>
    <img src='img/Screenshot from 2022-11-17 02-22-26.png'><br><img src='img/Screenshot from 2022-11-17 02-23-09.png'><br><img src='img/Screenshot from 2022-11-17 02-24-46.png'><br><p>Repeat this for every weight in the network using gradients from later layers</p>
    <h3>Neural Network in Practice: Optimization</h3>
    <img src='img/Screenshot from 2022-11-15 22-23-41.png'><br><h4>Loss Functions Can Be Difficult to Optimize</h4><br><br><h5>Remember:</h5><br><p>Optimization through gradient descent</p><br><img src='img/Screenshot from 2022-11-17 02-31-38.png'><br><img src='img/Screenshot from 2022-11-17 02-32-28.png'>
    <h4>Setting the Learning Rate</h4>
    <br><p>Small Lerning Rate converge and gets stuck in False local minima</p><br><br><p>Large Lerning Rate overshoot, become unstable and diverge</p><br><br><p>Stable Lerning Rate converge smoothly and avoid local minima</p><br>
    
<h4>How to deal with this?</h4>
    
<br><lh>Idea 1:</lh><li>Try lots of different learning rates and see what works \"just  right\" </li><br><br>

<lh>Idea 2:</lh><li>Do something smarter! <br> Design an adaptive learning rate that \"adapts\" to the landscape</li>
    <h4>Adaptive Learning Rates</h4>
    <br><lh>- Learnign Rate are not longer fixed </lh><br><br><lh>- Can be made larger or smaller depending on </lh><br><li>how large gradient is </li><br><li>How fast learning is happening</li><br><li>size of particular weights</li></li>
    
<h3>How to adjust learning rate using Pytorch</h3>
<p><u><i>torch.optim.lr_scheduler</i></u> provides several methods to adjust the learning rate based on the number of epochs.</p>
   
```python
  
  #define your model
  model = model
   
  #choose your optimizer
  optimizer = SGD(model.parameters(), 0.1)
   
  #define how you want to adjust your learnign rate
  scheduler = ExponentialLR(optimizer, gamma=0.9)
   
  training loop
  for epoch in range(20):
   
  #load datasets
     for input, target in dataset:
         #zero all of your previous gradients
         optimizer.zero_grad()
            
         #forward propagation
         output = model(input)
            
         #loss function
         loss = loss_fn(output, target)
            
         #back propagation
         loss.backward()
            
         #optimization
         optimizer.step()
            
        #update your learning rate after every epoch
       scheduler.step()  
 ```
    
 <h3>Gradient Descent Algorithms</h3>
 <br><li>SGD<i>{{ torch.optim.SGD(**args) }}</i></li><br>
 <li>Adam  <i>{{torch.optim.Adam(**args) }}</i></li>
 <br><li>Adafelta  <i>{{torch.optim.Adafelta(**args) }}</i></li>
 <br><li>Adagrad  <i>{{torch.optim.Adagrad(**args) }}</i></li>
 <br><li>RMSProp  <i>{{torch.optim.RMSProp(**args) }}</i></li>
   
 <h4>Putting it all together</h4>
    
   ```python 
    
    for input, target in dataset:
        #Clearning the old gradients from the last step
        optimizer.zero_grad()
   
        #Forward Propagation
        output = model(input)
   
        #calculation loss
        loss = loss_fn(output, target)
   
        #Calculating gradients of the loss w.r.t weights
        loss.backward()
   
        #Taking steps toward local minima
        optimizer.step()
   
   ```
    
 <h1>Now Let Make all of that make sense</h1>
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "2a55e7a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import libraries\n",
    "import torch\n",
    "import torchvision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7518a037",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "celltoolbar": "Slideshow",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

   
   




d text-normal pb-1">
                      Notified of all notifications on this repository.
                    </div>
                  </div>
                </button>

                <button
                  type="submit"
                  name="do"
                  value="ignore"
                  class="SelectMenu-item flex-items-start"
                  role="menuitemradio"
                  aria-checked="false"
                  data-targets="notifications-list-subscription-form.subscriptionButtons"
                >
                  <span class="f5">
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
                  </span>
                  <div>
                    <div class="f5 text-bold">
                      Ignore
                    </div>
                    <div class="text-small color-fg-muted text-normal pb-1">
                      Never be notified.
                    </div>
                  </div>
                </button>
</form>
              <button
                class="SelectMenu-item flex-items-start pr-3"
                type="button"
                role="menuitemradio"
                data-target="notifications-list-subscription-form.customButton"
                data-action="click:notifications-list-subscription-form#openCustomDialog"
                aria-haspopup="true"
                aria-checked="false"
                
              >
                <span class="f5">
                  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
                </span>
                <div>
                  <div class="d-flex flex-items-start flex-justify-between">
                    <div class="f5 text-bold">Custom</div>
                    <div class="f5 pr-1">
                      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-right">
    <path fill-rule="evenodd" d="M8.22 2.97a.75.75 0 011.06 0l4.25 4.25a.75.75 0 010 1.06l-4.25 4.25a.75.75 0 01-1.06-1.06l2.97-2.97H3.75a.75.75 0 010-1.5h7.44L8.22 4.03a.75.75 0 010-1.06z"></path>
</svg>
                    </div>
                  </div>
                  <div class="text-small color-fg-muted text-normal pb-1">
                    Select events you want to be notified of in addition to participating and @mentions.
                  </div>
                </div>
              </button>

            </div>
          </div>
        </details-menu>

        <details-dialog
          class="notifications-component-dialog "
          data-target="notifications-list-subscription-form.customDialog"
          aria-label="Custom dialog"
          hidden
        >
          <div class="SelectMenu-modal notifications-component-dialog-modal overflow-visible">
            <form data-target="notifications-list-subscription-form.customform" data-action="submit:notifications-list-subscription-form#submitCustomForm" data-turbo="false" action="/notifications/subscribe" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="fBE51y843c1oSff4-cIjepHGWg2Vq1nDa7JmDHTotZIFh29kUGq7golKx3E9MV0l1QBzaEOqzxa2z1NKS1tZ9g" autocomplete="off" />

              <input type="hidden" name="repository_id" value="15642233">

              <header class="d-sm-none SelectMenu-header pb-0 border-bottom-0 px-2 px-sm-3">
                <h1 class="f3 SelectMenu-title d-inline-flex">
                  <button
                    class="color-bg-default border-0 px-2 py-0 m-0 Link--secondary f5"
                    aria-label="Return to menu"
                    type="button"
                    data-action="click:notifications-list-subscription-form#closeCustomDialog"
                  >
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left">
    <path fill-rule="evenodd" d="M7.78 12.53a.75.75 0 01-1.06 0L2.47 8.28a.75.75 0 010-1.06l4.25-4.25a.75.75 0 011.06 1.06L4.81 7h7.44a.75.75 0 010 1.5H4.81l2.97 2.97a.75.75 0 010 1.06z"></path>
</svg>
                  </button>
                  Custom
                </h1>
              </header>

              <header class="d-none d-sm-flex flex-items-start pt-1">
                <button
                  class="border-0 px-2 pt-1 m-0 Link--secondary f5"
                  style="background-color: transparent;"
                  aria-label="Return to menu"
                  type="button"
                  data-action="click:notifications-list-subscription-form#closeCustomDialog"
                >
                  <svg style="position: relative; left: 2px; top: 1px" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left">
    <path fill-rule="evenodd" d="M7.78 12.53a.75.75 0 01-1.06 0L2.47 8.28a.75.75 0 010-1.06l4.25-4.25a.75.75 0 011.06 1.06L4.81 7h7.44a.75.75 0 010 1.5H4.81l2.97 2.97a.75.75 0 010 1.06z"></path>
</svg>
                </button>

                <h1 class="pt-1 pr-4 pb-0 pl-0 f5 text-bold">
                  Custom
                </h1>
              </header>

              <fieldset>
                <legend>
                  <div class="text-small color-fg-muted pt-0 pr-3 pb-3 pl-6 pl-sm-5 border-bottom mb-3">
                    Select events you want to be notified of in addition to participating and @mentions.
                  </div>
                </legend>
                <div data-target="notifications-list-subscription-form.labelInputs">
                </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="Issue"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                      >
                      Issues
                    </label>


                  </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="PullRequest"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                      >
                      Pull requests
                    </label>


                  </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="Release"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                      >
                      Releases
                    </label>


                  </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="Discussion"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                          aria-describedby="Discussion-disabled"
                          aria-disabled="true"
                      >
                      Discussions
                    </label>

                      <div
                        id="Discussion-disabled"
                        class="color-fg-muted"
                        >
                        Discussions are not enabled for this repository
                      </div>

                  </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="SecurityAlert"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                      >
                      Security alerts
                    </label>


                  </div>
              </fieldset>
              <div class="pt-2 pb-3 px-3 d-flex flex-justify-start flex-row-reverse">
                  <button name="do" value="custom" data-target="notifications-list-subscription-form.customSubmit" disabled="disabled" type="submit" data-view-component="true" class="btn-primary btn-sm btn ml-2">    Apply
</button>

                  <button data-action="click:notifications-list-subscription-form#resetForm" data-close-dialog="" type="button" data-view-component="true" class="btn-sm btn">    Cancel
</button>
              </div>
</form>          </div>
        </details-dialog>


        <div class="notifications-component-dialog-overlay"></div>
      </details>
    </notifications-list-subscription-form>



  </li>

  <li>
        <div data-view-component="true" class="BtnGroup">
        <a icon="repo-forked" href="/karpathy/convnetjs/fork" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;FORK_BUTTON&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="2b10f61c96a3086554ecc6c0f942b77e2c5693ce4effa5a1032f7116c735bfcb" data-ga-click="Repository, show fork modal, action:files#disambiguate; text:Fork" aria-label="Fork your own copy of karpathy/convnetjs" data-view-component="true" class="btn-sm btn BtnGroup-item">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo-forked mr-2">
    <path fill-rule="evenodd" d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"></path>
</svg>Fork
          <span id="repo-network-counter" data-pjax-replace="true" data-turbo-replace="true" title="2,034" data-view-component="true" class="Counter">2k</span>
</a>
      <details group_item="true" id="my-forks-menu-15642233" data-view-component="true" class="details-reset details-overlay BtnGroup-parent d-inline-block position-relative">
              <summary aria-label="See your forks of this repository" data-view-component="true" class="btn-sm btn BtnGroup-item px-2 float-none">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-triangle-down">
    <path d="M4.427 7.427l3.396 3.396a.25.25 0 00.354 0l3.396-3.396A.25.25 0 0011.396 7H4.604a.25.25 0 00-.177.427z"></path>
</svg>
</summary>
  <details-menu
    class="SelectMenu right-0"
      src="/karpathy/convnetjs/my_forks_menu_content?can_fork=true"
      
      role="menu"
      
>
    <div class="SelectMenu-modal">
        <button class="SelectMenu-closeButton position-absolute right-0 m-2" type="button" aria-label="Close menu" data-toggle-for="details-8475c0">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
</svg>
        </button>
      <div
        id="filter-menu-8475c0"
        class="d-flex flex-column flex-1 overflow-hidden"
>
        <div
          class="SelectMenu-list"
          >

            <include-fragment class="SelectMenu-loading" aria-label="Loading">
              <svg role="menuitem" style="box-sizing: content-box; color: var(--color-icon-primary);" width="32" height="32" viewBox="0 0 16 16" fill="none" data-view-component="true" class="anim-rotate">
  <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-opacity="0.25" stroke-width="2" vector-effect="non-scaling-stroke" />
  <path d="M15 8a7.002 7.002 0 00-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke" />
</svg>
            </include-fragment>
        </div>
        
      </div>
    </div>
  </details-menu>
</details></div>
  </li>

  <li>
        <template class="js-unstar-confirmation-dialog-template">
  <div class="Box-header">
    <h2 class="Box-title">Unstar this repository?</h2>
  </div>
  <div class="Box-body">
    <p class="mb-3">
      This will remove {{ repoNameWithOwner }} from the {{ listsWithCount }} that it's been added to.
    </p>
    <div class="form-actions">
      <form class="js-social-confirmation-form" data-turbo="false" action="{{ confirmUrl }}" accept-charset="UTF-8" method="post">
        <input type="hidden" name="authenticity_token" value="{{ confirmCsrfToken }}">
        <input type="hidden" name="confirm" value="true">
          <button data-close-dialog="true" type="submit" data-view-component="true" class="btn-danger btn width-full">    Unstar
</button>
</form>    </div>
  </div>
</template>

  <div data-view-component="true" class="js-toggler-container js-social-container starring-container d-flex">
    <div data-view-component="true" class="starred BtnGroup flex-1">
      <form class="js-social-form BtnGroup-parent flex-auto js-deferred-toggler-target" data-turbo="false" action="/karpathy/convnetjs/unstar" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="voUxnFVnDg3HJbvIGa7PXMcC6Pm_FEwz2vUD7x0RrW8VeWrz6CsVI9jmhf68ukJ3oV29HmOI6pGFaieGqnkuUg" autocomplete="off" />
          <input type="hidden" value="DWW3AKsHPiY51kNWqsygHWdXIXbk8dxPPptafeSUpYGmmexvFkslCCYVfWAP2C02AQh0kThteu1hBH4UU_wmvA" data-csrf="true" class="js-confirm-csrf-token" />
        <input type="hidden" name="context" value="repository">
          <button data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;UNSTAR_BUTTON&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="e8141d8540946eba74cdbedca13fa4b41bce5a303ed97425733a0f57db2febab" data-ga-click="Repository, click unstar button, action:files#disambiguate; text:Unstar" aria-label="Unstar this repository (10424)" type="submit" data-view-component="true" class="rounded-left-2 btn-sm btn BtnGroup-item">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star-fill starred-button-icon d-inline-block mr-2">
    <path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"></path>
</svg><span data-view-component="true" class="d-inline">
            Starred
</span>            <span id="repo-stars-counter-unstar" aria-label="10424 users starred this repository" data-singular-suffix="user starred this repository" data-plural-suffix="users starred this repository" data-turbo-replace="true" title="10,424" data-view-component="true" class="Counter js-social-count">10.4k</span>
</button></form>        <details id="details-user-list-15642233" data-view-component="true" class="details-reset details-overlay BtnGroup-parent js-user-list-menu d-inline-block position-relative">
        <summary aria-label="Add this repository to a list" data-view-component="true" class="btn-sm btn BtnGroup-item px-2 float-none">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-triangle-down">
    <path d="M4.427 7.427l3.396 3.396a.25.25 0 00.354 0l3.396-3.396A.25.25 0 0011.396 7H4.604a.25.25 0 00-.177.427z"></path>
</svg>
</summary>
  <details-menu
    class="SelectMenu right-0"
      src="/karpathy/convnetjs/lists"
      
      role="menu"
      
>
    <div class="SelectMenu-modal">
        <button class="SelectMenu-closeButton position-absolute right-0 m-2" type="button" aria-label="Close menu" data-toggle-for="details-da9d5b">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
</svg>
        </button>
      <div
        id="filter-menu-da9d5b"
        class="d-flex flex-column flex-1 overflow-hidden"
>
        <div
          class="SelectMenu-list"
          >

            <include-fragment class="SelectMenu-loading" aria-label="Loading">
              <svg role="menuitem" style="box-sizing: content-box; color: var(--color-icon-primary);" width="32" height="32" viewBox="0 0 16 16" fill="none" data-view-component="true" class="anim-rotate">
  <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-opacity="0.25" stroke-width="2" vector-effect="non-scaling-stroke" />
  <path d="M15 8a7.002 7.002 0 00-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke" />
</svg>
            </include-fragment>
        </div>
        
      </div>
    </div>
  </details-menu>
</details>
</div>
    <div data-view-component="true" class="unstarred BtnGroup flex-1">
      <form class="js-social-form BtnGroup-parent flex-auto" data-turbo="false" action="/karpathy/convnetjs/star" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="H4z9HdM4rLsF4wYPLCIVsHME8jAnQwZPpOualkQYRDpvuGQMhbgxt6w1ywMpHi-Q-HARsIc0vhXdyPpzyZz4yw" autocomplete="off" />
        <input type="hidden" name="context" value="repository">
          <button data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;STAR_BUTTON&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="398f45ab442c0dac7fd68a43dc530f905f8de778c3acecd5a3901e0731aed1ee" data-ga-click="Repository, click star button, action:files#disambiguate; text:Star" aria-label="Star this repository (10424)" type="submit" data-view-component="true" class="js-toggler-target rounded-left-2 btn-sm btn BtnGroup-item">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star d-inline-block mr-2">
    <path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25zm0 2.445L6.615 5.5a.75.75 0 01-.564.41l-3.097.45 2.24 2.184a.75.75 0 01.216.664l-.528 3.084 2.769-1.456a.75.75 0 01.698 0l2.77 1.456-.53-3.084a.75.75 0 01.216-.664l2.24-2.183-3.096-.45a.75.75 0 01-.564-.41L8 2.694v.001z"></path>
</svg><span data-view-component="true" class="d-inline">
            Star
</span>            <span id="repo-stars-counter-star" aria-label="10424 users starred this repository" data-singular-suffix="user starred this repository" data-plural-suffix="users starred this repository" data-turbo-replace="true" title="10,424" data-view-component="true" class="Counter js-social-count">10.4k</span>
</button></form>        <details id="details-user-list-15642233" data-view-component="true" class="details-reset details-overlay BtnGroup-parent js-user-list-menu d-inline-block position-relative">
        <summary aria-label="Add this repository to a list" data-view-component="true" class="btn-sm btn BtnGroup-item px-2 float-none">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-triangle-down">
    <path d="M4.427 7.427l3.396 3.396a.25.25 0 00.354 0l3.396-3.396A.25.25 0 0011.396 7H4.604a.25.25 0 00-.177.427z"></path>
</svg>
</summary>
  <details-menu
    class="SelectMenu right-0"
      src="/karpathy/convnetjs/lists"
      
      role="menu"
      
>
    <div class="SelectMenu-modal">
        <button class="SelectMenu-closeButton position-absolute right-0 m-2" type="button" aria-label="Close menu" data-toggle-for="details-da9d5b">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
</svg>
        </button>
      <div
        id="filter-menu-da9d5b"
        class="d-flex flex-column flex-1 overflow-hidden"
>
        <div
          class="SelectMenu-list"
          >

            <include-fragment class="SelectMenu-loading" aria-label="Loading">
              <svg role="menuitem" style="box-sizing: content-box; color: var(--color-icon-primary);" width="32" height="32" viewBox="0 0 16 16" fill="none" data-view-component="true" class="anim-rotate">
  <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-opacity="0.25" stroke-width="2" vector-effect="non-scaling-stroke" />
  <path d="M15 8a7.002 7.002 0 00-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke" />
</svg>
            </include-fragment>
        </div>
        
      </div>
    </div>
  </details-menu>
</details>
</div></div>
  </li>

    

</ul>

      </div>

        <div id="responsive-meta-container" data-turbo-replace>
      <div class="d-block d-md-none mb-2 px-3 px-md-4 px-lg-5">
      <p class="f4 mb-3 ">
        Deep Learning in Javascript. Train Convolutional Neural Networks (or ordinary ones) in your browser.
      </p>

    <h3 class="sr-only">License</h3>
  <div class="mb-2">
    <a href="/karpathy/convnetjs/blob/master/LICENSE"
      class="Link--muted"
      
      data-analytics-event="{&quot;category&quot;:&quot;Repository Overview&quot;,&quot;action&quot;:&quot;click&quot;,&quot;label&quot;:&quot;location:sidebar;file:license&quot;}"
    >
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-law mr-2">
    <path fill-rule="evenodd" d="M8.75.75a.75.75 0 00-1.5 0V2h-.984c-.305 0-.604.08-.869.23l-1.288.737A.25.25 0 013.984 3H1.75a.75.75 0 000 1.5h.428L.066 9.192a.75.75 0 00.154.838l.53-.53-.53.53v.001l.002.002.002.002.006.006.016.015.045.04a3.514 3.514 0 00.686.45A4.492 4.492 0 003 11c.88 0 1.556-.22 2.023-.454a3.515 3.515 0 00.686-.45l.045-.04.016-.015.006-.006.002-.002.001-.002L5.25 9.5l.53.53a.75.75 0 00.154-.838L3.822 4.5h.162c.305 0 .604-.08.869-.23l1.289-.737a.25.25 0 01.124-.033h.984V13h-2.5a.75.75 0 000 1.5h6.5a.75.75 0 000-1.5h-2.5V3.5h.984a.25.25 0 01.124.033l1.29.736c.264.152.563.231.868.231h.162l-2.112 4.692a.75.75 0 00.154.838l.53-.53-.53.53v.001l.002.002.002.002.006.006.016.015.045.04a3.517 3.517 0 00.686.45A4.492 4.492 0 0013 11c.88 0 1.556-.22 2.023-.454a3.512 3.512 0 00.686-.45l.045-.04.01-.01.006-.005.006-.006.002-.002.001-.002-.529-.531.53.53a.75.75 0 00.154-.838L13.823 4.5h.427a.75.75 0 000-1.5h-2.234a.25.25 0 01-.124-.033l-1.29-.736A1.75 1.75 0 009.735 2H8.75V.75zM1.695 9.227c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L3 6.327l-1.305 2.9zm10 0c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L13 6.327l-1.305 2.9z"></path>
</svg>
     MIT license
    </a>
  </div>


    <div class="mb-3">
        <a class="Link--secondary no-underline mr-3" href="/karpathy/convnetjs/stargazers">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star mr-1">
    <path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25zm0 2.445L6.615 5.5a.75.75 0 01-.564.41l-3.097.45 2.24 2.184a.75.75 0 01.216.664l-.528 3.084 2.769-1.456a.75.75 0 01.698 0l2.77 1.456-.53-3.084a.75.75 0 01.216-.664l2.24-2.183-3.096-.45a.75.75 0 01-.564-.41L8 2.694v.001z"></path>
</svg>
          <span class="text-bold">10.4k</span>
          stars
</a>        <a class="Link--secondary no-underline" href="/karpathy/convnetjs/network/members">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo-forked mr-1">
    <path fill-rule="evenodd" d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"></path>
</svg>
          <span class="text-bold">2k</span>
          forks
</a>    </div>

      <div class="d-flex">
        <div class="flex-1 mr-2">
            <template class="js-unstar-confirmation-dialog-template">
  <div class="Box-header">
    <h2 class="Box-title">Unstar this repository?</h2>
  </div>
  <div class="Box-body">
    <p class="mb-3">
      This will remove {{ repoNameWithOwner }} from the {{ listsWithCount }} that it's been added to.
    </p>
    <div class="form-actions">
      <form class="js-social-confirmation-form" data-turbo="false" action="{{ confirmUrl }}" accept-charset="UTF-8" method="post">
        <input type="hidden" name="authenticity_token" value="{{ confirmCsrfToken }}">
        <input type="hidden" name="confirm" value="true">
          <button data-close-dialog="true" type="submit" data-view-component="true" class="btn-danger btn width-full">    Unstar
</button>
</form>    </div>
  </div>
</template>

  <div data-view-component="true" class="js-toggler-container js-social-container starring-container d-flex">
    <div data-view-component="true" class="starred BtnGroup flex-1">
      <form class="js-social-form BtnGroup-parent flex-auto width-full js-deferred-toggler-target" data-turbo="false" action="/karpathy/convnetjs/unstar" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="2rKJn7Wzkzn3lQVxhMszwcGMOjxRKmqgq1u27x_10SJxTtLwCP-IF-hWO0ch377qp9Nv2422zAL0xJKGqJ1SHw" autocomplete="off" />
          <input type="hidden" value="-Oa_JqPclm69E2q35-aPQKj6bnFA0Fow-chpSjdPjAlTGuRJHpCNQKLQVIFC8gJrzqU7lpxM_JKmV00jgCcPNA" data-csrf="true" class="js-confirm-csrf-token" />
        <input type="hidden" name="context" value="repository">
          <button data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;UNSTAR_BUTTON&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="e8141d8540946eba74cdbedca13fa4b41bce5a303ed97425733a0f57db2febab" data-ga-click="Repository, click unstar button, action:files#disambiguate; text:Unstar" aria-label="Unstar this repository" type="submit" data-view-component="true" class="rounded-left-2 btn-sm btn btn-block BtnGroup-item">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star-fill starred-button-icon d-inline-block mr-2">
    <path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"></path>
</svg><span data-view-component="true" class="d-inline">
            Starred
</span>
</button></form></div>
    <div data-view-component="true" class="unstarred BtnGroup flex-1">
      <form class="js-social-form BtnGroup-parent flex-auto width-full" data-turbo="false" action="/karpathy/convnetjs/star" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="fSLzMIRMTshKp5-Ck_7685wFFGQJMtcogaopWcu6PwENFmoh0szTxONxUo6WwsDTF3H35KlFb3L4iUm8Rj6D8A" autocomplete="off" />
        <input type="hidden" name="context" value="repository">
          <button data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;STAR_BUTTON&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="398f45ab442c0dac7fd68a43dc530f905f8de778c3acecd5a3901e0731aed1ee" data-ga-click="Repository, click star button, action:files#disambiguate; text:Star" aria-label="Star this repository" type="submit" data-view-component="true" class="js-toggler-target rounded-left-2 btn-sm btn btn-block BtnGroup-item">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star d-inline-block mr-2">
    <path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25zm0 2.445L6.615 5.5a.75.75 0 01-.564.41l-3.097.45 2.24 2.184a.75.75 0 01.216.664l-.528 3.084 2.769-1.456a.75.75 0 01.698 0l2.77 1.456-.53-3.084a.75.75 0 01.216-.664l2.24-2.183-3.096-.45a.75.75 0 01-.564-.41L8 2.694v.001z"></path>
</svg><span data-view-component="true" class="d-inline">
            Star
</span>
</button></form></div></div>
        </div>
        <div class="flex-1">
              <notifications-list-subscription-form
      data-action="notifications-dialog-label-toggled:notifications-list-subscription-form#handleDialogLabelToggle"
      class="f5 position-relative"
    >
      <details
        class="details-reset details-overlay f5 position-relative"
        data-target="notifications-list-subscription-form.details"
        data-action="toggle:notifications-list-subscription-form#detailsToggled"
      >

        <summary data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;WATCH_BUTTON&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="a2d4ede252f7d7aacd23e13e993f31532a1d2279137cc9bc6ebcd5975dc46735" data-ga-click="Repository, click Watch settings, action:files#disambiguate" aria-label="Notification settings" data-view-component="true" class="btn-sm btn btn-block">    <span data-menu-button>
            <span
              hidden
              
              data-target="notifications-list-subscription-form.unwatchButtonCopy"
            >
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-eye">
    <path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path>
</svg>
              Unwatch
            </span>
            <span
              hidden
              
              data-target="notifications-list-subscription-form.stopIgnoringButtonCopy"
            >
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-bell-slash">
    <path fill-rule="evenodd" d="M8 1.5c-.997 0-1.895.416-2.534 1.086A.75.75 0 014.38 1.55 5 5 0 0113 5v2.373a.75.75 0 01-1.5 0V5A3.5 3.5 0 008 1.5zM4.182 4.31L1.19 2.143a.75.75 0 10-.88 1.214L3 5.305v2.642a.25.25 0 01-.042.139L1.255 10.64A1.518 1.518 0 002.518 13h11.108l1.184.857a.75.75 0 10.88-1.214l-1.375-.996a1.196 1.196 0 00-.013-.01L4.198 4.321a.733.733 0 00-.016-.011zm7.373 7.19L4.5 6.391v1.556c0 .346-.102.683-.294.97l-1.703 2.556a.018.018 0 00-.003.01.015.015 0 00.005.012.017.017 0 00.006.004l.007.001h9.037zM8 16a2 2 0 001.985-1.75c.017-.137-.097-.25-.235-.25h-3.5c-.138 0-.252.113-.235.25A2 2 0 008 16z"></path>
</svg>
              Stop ignoring
            </span>
            <span
              
              
              data-target="notifications-list-subscription-form.watchButtonCopy"
            >
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-eye">
    <path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path>
</svg>
              Watch
            </span>
          </span>
          <span class="dropdown-caret"></span>
</summary>
        <details-menu
          class="SelectMenu  "
          role="menu"
          data-target="notifications-list-subscription-form.menu"
          
        >
          <div class="SelectMenu-modal notifications-component-menu-modal">
            <header class="SelectMenu-header">
              <h3 class="SelectMenu-title">Notifications</h3>
              <button class="SelectMenu-closeButton" type="button" aria-label="Close menu" data-action="click:notifications-list-subscription-form#closeMenu">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
</svg>
              </button>
            </header>

            <div class="SelectMenu-list">
              <form data-target="notifications-list-subscription-form.form" data-action="submit:notifications-list-subscription-form#submitForm" data-turbo="false" action="/notifications/subscribe" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="Z8FdJNS0eJfEZM0TUn6fXz4ImqeNT7VAAqdT-jn2dXseVwuXq-Ye2CVn_ZqWjeEAes6zwltOI5Xf2ma8BkWZHw" autocomplete="off" />

                <input type="hidden" name="repository_id" value="15642233">

                <button
                  type="submit"
                  name="do"
                  value="included"
                  class="SelectMenu-item flex-items-start"
                  role="menuitemradio"
                  aria-checked="true"
                  data-targets="notifications-list-subscription-form.subscriptionButtons"
                  
                >
                  <span class="f5">
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
                  </span>
                  <div>
                    <div class="f5 text-bold">
                      Participating and @mentions
                    </div>
                    <div class="text-small color-fg-muted text-normal pb-1">
                      Only receive notifications from this repository when participating or @mentioned.
                    </div>
                  </div>
                </button>

                <button
                  type="submit"
                  name="do"
                  value="subscribed"
                  class="SelectMenu-item flex-items-start"
                  role="menuitemradio"
                  aria-checked="false"
                  data-targets="notifications-list-subscription-form.subscriptionButtons"
                >
                  <span class="f5">
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
                  </span>
                  <div>
                    <div class="f5 text-bold">
                      All Activity
                    </div>
                    <div class="text-small color-fg-muted text-normal pb-1">
                      Notified of all notifications on this repository.
                    </div>
                  </div>
                </button>

                <button
                  type="submit"
                  name="do"
                  value="ignore"
                  class="SelectMenu-item flex-items-start"
                  role="menuitemradio"
                  aria-checked="false"
                  data-targets="notifications-list-subscription-form.subscriptionButtons"
                >
                  <span class="f5">
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
                  </span>
                  <div>
                    <div class="f5 text-bold">
                      Ignore
                    </div>
                    <div class="text-small color-fg-muted text-normal pb-1">
                      Never be notified.
                    </div>
                  </div>
                </button>
</form>
              <button
                class="SelectMenu-item flex-items-start pr-3"
                type="button"
                role="menuitemradio"
                data-target="notifications-list-subscription-form.customButton"
                data-action="click:notifications-list-subscription-form#openCustomDialog"
                aria-haspopup="true"
                aria-checked="false"
                
              >
                <span class="f5">
                  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
                </span>
                <div>
                  <div class="d-flex flex-items-start flex-justify-between">
                    <div class="f5 text-bold">Custom</div>
                    <div class="f5 pr-1">
                      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-right">
    <path fill-rule="evenodd" d="M8.22 2.97a.75.75 0 011.06 0l4.25 4.25a.75.75 0 010 1.06l-4.25 4.25a.75.75 0 01-1.06-1.06l2.97-2.97H3.75a.75.75 0 010-1.5h7.44L8.22 4.03a.75.75 0 010-1.06z"></path>
</svg>
                    </div>
                  </div>
                  <div class="text-small color-fg-muted text-normal pb-1">
                    Select events you want to be notified of in addition to participating and @mentions.
                  </div>
                </div>
              </button>

            </div>
          </div>
        </details-menu>

        <details-dialog
          class="notifications-component-dialog "
          data-target="notifications-list-subscription-form.customDialog"
          aria-label="Custom dialog"
          hidden
        >
          <div class="SelectMenu-modal notifications-component-dialog-modal overflow-visible">
            <form data-target="notifications-list-subscription-form.customform" data-action="submit:notifications-list-subscription-form#submitCustomForm" data-turbo="false" action="/notifications/subscribe" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="38bZvwT-pyfnh3jgoN2EWoXLeYec774xMNcyVN_jy9CmUI8Me6zBaAaESGlkLvoFwQ1Q4kruKOTtqgcS4FAntA" autocomplete="off" />

              <input type="hidden" name="repository_id" value="15642233">

              <header class="d-sm-none SelectMenu-header pb-0 border-bottom-0 px-2 px-sm-3">
                <h1 class="f3 SelectMenu-title d-inline-flex">
                  <button
                    class="color-bg-default border-0 px-2 py-0 m-0 Link--secondary f5"
                    aria-label="Return to menu"
                    type="button"
                    data-action="click:notifications-list-subscription-form#closeCustomDialog"
                  >
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left">
    <path fill-rule="evenodd" d="M7.78 12.53a.75.75 0 01-1.06 0L2.47 8.28a.75.75 0 010-1.06l4.25-4.25a.75.75 0 011.06 1.06L4.81 7h7.44a.75.75 0 010 1.5H4.81l2.97 2.97a.75.75 0 010 1.06z"></path>
</svg>
                  </button>
                  Custom
                </h1>
              </header>

              <header class="d-none d-sm-flex flex-items-start pt-1">
                <button
                  class="border-0 px-2 pt-1 m-0 Link--secondary f5"
                  style="background-color: transparent;"
                  aria-label="Return to menu"
                  type="button"
                  data-action="click:notifications-list-subscription-form#closeCustomDialog"
                >
                  <svg style="position: relative; left: 2px; top: 1px" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left">
    <path fill-rule="evenodd" d="M7.78 12.53a.75.75 0 01-1.06 0L2.47 8.28a.75.75 0 010-1.06l4.25-4.25a.75.75 0 011.06 1.06L4.81 7h7.44a.75.75 0 010 1.5H4.81l2.97 2.97a.75.75 0 010 1.06z"></path>
</svg>
                </button>

                <h1 class="pt-1 pr-4 pb-0 pl-0 f5 text-bold">
                  Custom
                </h1>
              </header>

              <fieldset>
                <legend>
                  <div class="text-small color-fg-muted pt-0 pr-3 pb-3 pl-6 pl-sm-5 border-bottom mb-3">
                    Select events you want to be notified of in addition to participating and @mentions.
                  </div>
                </legend>
                <div data-target="notifications-list-subscription-form.labelInputs">
                </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="Issue"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                      >
                      Issues
                    </label>


                  </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="PullRequest"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                      >
                      Pull requests
                    </label>


                  </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="Release"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                      >
                      Releases
                    </label>


                  </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="Discussion"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                          aria-describedby="Discussion-disabled"
                          aria-disabled="true"
                      >
                      Discussions
                    </label>

                      <div
                        id="Discussion-disabled"
                        class="color-fg-muted"
                        >
                        Discussions are not enabled for this repository
                      </div>

                  </div>
                  <div class="form-checkbox mr-3 ml-6 ml-sm-5 mb-2 mt-0">
                    <label class="f5 text-normal">
                      <input
                        type="checkbox"
                        name="thread_types[]"
                        value="SecurityAlert"
                        data-targets="notifications-list-subscription-form.threadTypeCheckboxes"
                        data-action="change:notifications-list-subscription-form#threadTypeCheckboxesUpdated"
                        
                      >
                      Security alerts
                    </label>


                  </div>
              </fieldset>
              <div class="pt-2 pb-3 px-3 d-flex flex-justify-start flex-row-reverse">
                  <button name="do" value="custom" data-target="notifications-list-subscription-form.customSubmit" disabled="disabled" type="submit" data-view-component="true" class="btn-primary btn-sm btn ml-2">    Apply
</button>

                  <button data-action="click:notifications-list-subscription-form#resetForm" data-close-dialog="" type="button" data-view-component="true" class="btn-sm btn">    Cancel
</button>
              </div>
</form>          </div>
        </details-dialog>


        <div class="notifications-component-dialog-overlay"></div>
      </details>
    </notifications-list-subscription-form>



        </div>
      </div>
  </div>

</div>


          <nav data-pjax="#js-repo-pjax-container" aria-label="Repository" data-view-component="true" class="js-repo-nav js-sidenav-container-pjax js-responsive-underlinenav overflow-hidden UnderlineNav px-3 px-md-4 px-lg-5">

  <ul data-view-component="true" class="UnderlineNav-body list-style-none">
      <li data-view-component="true" class="d-inline-flex">
  <a id="code-tab" href="/karpathy/convnetjs" data-tab-item="i0code-tab" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages repo_deployments /karpathy/convnetjs" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g c" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Code&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" aria-current="page" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item selected">
    
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-code UnderlineNav-octicon d-none d-sm-inline">
    <path fill-rule="evenodd" d="M4.72 3.22a.75.75 0 011.06 1.06L2.06 8l3.72 3.72a.75.75 0 11-1.06 1.06L.47 8.53a.75.75 0 010-1.06l4.25-4.25zm6.56 0a.75.75 0 10-1.06 1.06L13.94 8l-3.72 3.72a.75.75 0 101.06 1.06l4.25-4.25a.75.75 0 000-1.06l-4.25-4.25z"></path>
</svg>
        <span data-content="Code">Code</span>
          <span id="code-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="issues-tab" href="/karpathy/convnetjs/issues" data-tab-item="i1issues-tab" data-selected-links="repo_issues repo_labels repo_milestones /karpathy/convnetjs/issues" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g i" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Issues&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-opened UnderlineNav-octicon d-none d-sm-inline">
    <path d="M8 9.5a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path><path fill-rule="evenodd" d="M8 0a8 8 0 100 16A8 8 0 008 0zM1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0z"></path>
</svg>
        <span data-content="Issues">Issues</span>
          <span id="issues-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="51" data-view-component="true" class="Counter">51</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="pull-requests-tab" href="/karpathy/convnetjs/pulls" data-tab-item="i2pull-requests-tab" data-selected-links="repo_pulls checks /karpathy/convnetjs/pulls" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g p" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Pull requests&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request UnderlineNav-octicon d-none d-sm-inline">
    <path fill-rule="evenodd" d="M7.177 3.073L9.573.677A.25.25 0 0110 .854v4.792a.25.25 0 01-.427.177L7.177 3.427a.25.25 0 010-.354zM3.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122v5.256a2.251 2.251 0 11-1.5 0V5.372A2.25 2.25 0 011.5 3.25zM11 2.5h-1V4h1a1 1 0 011 1v5.628a2.251 2.251 0 101.5 0V5A2.5 2.5 0 0011 2.5zm1 10.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0zM3.75 12a.75.75 0 100 1.5.75.75 0 000-1.5z"></path>
</svg>
        <span data-content="Pull requests">Pull requests</span>
          <span id="pull-requests-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="23" data-view-component="true" class="Counter">23</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="actions-tab" href="/karpathy/convnetjs/actions" data-tab-item="i3actions-tab" data-selected-links="repo_actions /karpathy/convnetjs/actions" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g a" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Actions&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-play UnderlineNav-octicon d-none d-sm-inline">
    <path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM6.379 5.227A.25.25 0 006 5.442v5.117a.25.25 0 00.379.214l4.264-2.559a.25.25 0 000-.428L6.379 5.227z"></path>
</svg>
        <span data-content="Actions">Actions</span>
          <span id="actions-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="projects-tab" href="/karpathy/convnetjs/projects" data-tab-item="i4projects-tab" data-selected-links="repo_projects new_repo_project repo_project /karpathy/convnetjs/projects" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g b" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Projects&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-table UnderlineNav-octicon d-none d-sm-inline">
    <path fill-rule="evenodd" d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0114.25 16H1.75A1.75 1.75 0 010 14.25V1.75zM1.5 6.5v7.75c0 .138.112.25.25.25H5v-8H1.5zM5 5H1.5V1.75a.25.25 0 01.25-.25H5V5zm1.5 1.5v8h7.75a.25.25 0 00.25-.25V6.5h-8zm8-1.5h-8V1.5h7.75a.25.25 0 01.25.25V5z"></path>
</svg>
        <span data-content="Projects">Projects</span>
          <span id="projects-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="0" hidden="hidden" data-view-component="true" class="Counter">0</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="wiki-tab" href="/karpathy/convnetjs/wiki" data-tab-item="i5wiki-tab" data-selected-links="repo_wiki /karpathy/convnetjs/wiki" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g w" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Wiki&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-book UnderlineNav-octicon d-none d-sm-inline">
    <path fill-rule="evenodd" d="M0 1.75A.75.75 0 01.75 1h4.253c1.227 0 2.317.59 3 1.501A3.744 3.744 0 0111.006 1h4.245a.75.75 0 01.75.75v10.5a.75.75 0 01-.75.75h-4.507a2.25 2.25 0 00-1.591.659l-.622.621a.75.75 0 01-1.06 0l-.622-.621A2.25 2.25 0 005.258 13H.75a.75.75 0 01-.75-.75V1.75zm8.755 3a2.25 2.25 0 012.25-2.25H14.5v9h-3.757c-.71 0-1.4.201-1.992.572l.004-7.322zm-1.504 7.324l.004-5.073-.002-2.253A2.25 2.25 0 005.003 2.5H1.5v9h3.757a3.75 3.75 0 011.994.574z"></path>
</svg>
        <span data-content="Wiki">Wiki</span>
          <span id="wiki-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="security-tab" href="/karpathy/convnetjs/security" data-tab-item="i6security-tab" data-selected-links="security overview alerts policy token_scanning code_scanning /karpathy/convnetjs/security" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g s" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Security&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-shield UnderlineNav-octicon d-none d-sm-inline">
    <path fill-rule="evenodd" d="M7.467.133a1.75 1.75 0 011.066 0l5.25 1.68A1.75 1.75 0 0115 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.7 1.7 0 01-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 011.217-1.667l5.25-1.68zm.61 1.429a.25.25 0 00-.153 0l-5.25 1.68a.25.25 0 00-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.2.2 0 00.154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.25.25 0 00-.174-.237l-5.25-1.68zM9 10.5a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.75a.75.75 0 10-1.5 0v3a.75.75 0 001.5 0v-3z"></path>
</svg>
        <span data-content="Security">Security</span>
          <include-fragment src="/karpathy/convnetjs/security/overall-count" accept="text/fragment+html"></include-fragment>

    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="insights-tab" href="/karpathy/convnetjs/pulse" data-tab-item="i7insights-tab" data-selected-links="repo_graphs repo_contributors dependency_graph dependabot_updates pulse people community /karpathy/convnetjs/pulse" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Insights&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-graph UnderlineNav-octicon d-none d-sm-inline">
    <path fill-rule="evenodd" d="M1.5 1.75a.75.75 0 00-1.5 0v12.5c0 .414.336.75.75.75h14.5a.75.75 0 000-1.5H1.5V1.75zm14.28 2.53a.75.75 0 00-1.06-1.06L10 7.94 7.53 5.47a.75.75 0 00-1.06 0L3.22 8.72a.75.75 0 001.06 1.06L7 7.06l2.47 2.47a.75.75 0 001.06 0l5.25-5.25z"></path>
</svg>
        <span data-content="Insights">Insights</span>
          <span id="insights-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
</ul>
    <div style="visibility:hidden;" data-view-component="true" class="UnderlineNav-actions js-responsive-underlinenav-overflow position-absolute pr-3 pr-md-4 pr-lg-5 right-0">      <details data-view-component="true" class="details-overlay details-reset position-relative">
  <summary role="button" data-view-component="true">          <div class="UnderlineNav-item mr-0 border-0">
            <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-kebab-horizontal">
    <path d="M8 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm13 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path>
</svg>
            <span class="sr-only">More</span>
          </div>
</summary>
  <details-menu role="menu" data-view-component="true" class="dropdown-menu dropdown-menu-sw">          <ul>
              <li data-menu-item="i0code-tab" hidden>
                <a role="menuitem" class="js-selected-navigation-item selected dropdown-item" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages repo_deployments /karpathy/convnetjs" href="/karpathy/convnetjs">
                  Code
</a>              </li>
              <li data-menu-item="i1issues-tab" hidden>
                <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links="repo_issues repo_labels repo_milestones /karpathy/convnetjs/issues" href="/karpathy/convnetjs/issues">
                  Issues
</a>              </li>
              <li data-menu-item="i2pull-requests-tab" hidden>
                <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links="repo_pulls checks /karpathy/convnetjs/pulls" href="/karpathy/convnetjs/pulls">
                  Pull requests
</a>              </li>
              <li data-menu-item="i3actions-tab" hidden>
                <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links="repo_actions /karpathy/convnetjs/actions" href="/karpathy/convnetjs/actions">
                  Actions
</a>              </li>
              <li data-menu-item="i4projects-tab" hidden>
                <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links="repo_projects new_repo_project repo_project /karpathy/convnetjs/projects" href="/karpathy/convnetjs/projects">
                  Projects
</a>              </li>
              <li data-menu-item="i5wiki-tab" hidden>
                <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links="repo_wiki /karpathy/convnetjs/wiki" href="/karpathy/convnetjs/wiki">
                  Wiki
</a>              </li>
              <li data-menu-item="i6security-tab" hidden>
                <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links="security overview alerts policy token_scanning code_scanning /karpathy/convnetjs/security" href="/karpathy/convnetjs/security">
                  Security
</a>              </li>
              <li data-menu-item="i7insights-tab" hidden>
                <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links="repo_graphs repo_contributors dependency_graph dependabot_updates pulse people community /karpathy/convnetjs/pulse" href="/karpathy/convnetjs/pulse">
                  Insights
</a>              </li>
          </ul>
</details-menu>
</details></div>
</nav>



  </div>



  <turbo-frame id="repo-content-turbo-frame" target="_top" data-turbo-action="advance" class="">
      <div id="repo-content-pjax-container" class="repository-content " >
      <a href="https://github.dev/" class="d-none js-github-dev-shortcut" data-hotkey=".">Open in github.dev</a>
  <a href="https://github.dev/" class="d-none js-github-dev-new-tab-shortcut" data-hotkey="Shift+.,Shift+&gt;,&gt;" target="_blank">Open in a new github.dev tab</a>



    
      
  <h1 class='sr-only'>karpathy/convnetjs</h1>
  <div class="clearfix container-xl px-3 px-md-4 px-lg-5 mt-4">
    

<div>
  

  <div class="d-none d-lg-block mt-6 mr-3 Popover top-0 right-0 color-shadow-medium col-3">
    
  </div>

  <div id="spoof-warning" class="mt-0 pb-3" hidden aria-hidden>
  <div data-view-component="true" class="flash flash-warn mt-0 clearfix">
  
  
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-alert float-left mt-1">
    <path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path>
</svg>

      <div class="overflow-hidden">This commit does not belong to any branch on this repository, and may belong to a fork outside of the repository.</div>


  
</div></div>

  <include-fragment src="/karpathy/convnetjs/spoofed_commit_check/4c3358a315b4d71f31a0d532eb5d1700e9e592ee" data-test-selector="spoofed-commit-check"></include-fragment>

  <div data-view-component="true" class="Layout Layout--flowRow-until-md Layout--sidebarPosition-end Layout--sidebarPosition-flowRow-end">
  <div data-view-component="true" class="Layout-main">      
      
        <include-fragment src="/karpathy/convnetjs/show_partial?partial=tree%2Frecently_touched_branches_list"></include-fragment>
      <div class="file-navigation mb-3 d-flex flex-items-start">
  
<div class="position-relative">
  <details
    class="js-branch-select-menu details-reset details-overlay mr-0 mb-0 "
    id="branch-select-menu"
    data-hydro-click-payload="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;REFS_SELECTOR_MENU&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="c2cd22b614d9f529e6a9f83e88d71ccce124ee8743ead158a3ffb043929b8299">
    <summary class="btn css-truncate"
            data-hotkey="w"
            title="Switch branches or tags">
      <svg text="gray" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-branch">
    <path fill-rule="evenodd" d="M11.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122V6A2.5 2.5 0 0110 8.5H6a1 1 0 00-1 1v1.128a2.251 2.251 0 11-1.5 0V5.372a2.25 2.25 0 111.5 0v1.836A2.492 2.492 0 016 7h4a1 1 0 001-1v-.628A2.25 2.25 0 019.5 3.25zM4.25 12a.75.75 0 100 1.5.75.75 0 000-1.5zM3.5 3.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0z"></path>
</svg>
      <span class="css-truncate-target" data-menu-button>master</span>
      <span class="dropdown-caret"></span>
    </summary>

    
<div class="SelectMenu">
  <div class="SelectMenu-modal">
    <header class="SelectMenu-header">
      <span class="SelectMenu-title">Switch branches/tags</span>
      <button class="SelectMenu-closeButton" type="button" data-toggle-for="branch-select-menu"><svg aria-label="Close menu" aria-hidden="false" role="img" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
</svg></button>
    </header>

    <input-demux data-action="tab-container-change:input-demux#storeInput tab-container-changed:input-demux#updateInput">
      <tab-container class="d-flex flex-column js-branches-tags-tabs" style="min-height: 0;">
        <div class="SelectMenu-filter">
          <input data-target="input-demux.source"
                 id="context-commitish-filter-field"
                 class="SelectMenu-input form-control"
                 aria-owns="ref-list-branches"
                 data-controls-ref-menu-id="ref-list-branches"
                 autofocus
                 autocomplete="off"
                 aria-label="Filter branches/tags"
                 placeholder="Filter branches/tags"
                 type="text"
          >
        </div>

        <div class="SelectMenu-tabs" role="tablist" data-target="input-demux.control" >
          <button class="SelectMenu-tab" type="button" role="tab" aria-selected="true">Branches</button>
          <button class="SelectMenu-tab" type="button" role="tab">Tags</button>
        </div>

        <div role="tabpanel" id="ref-list-branches" data-filter-placeholder="Filter branches/tags" tabindex="" class="d-flex flex-column flex-auto overflow-auto">
          <ref-selector
            type="branch"
            data-targets="input-demux.sinks"
            data-action="
              input-entered:ref-selector#inputEntered
              tab-selected:ref-selector#tabSelected
              focus-list:ref-selector#focusFirstListMember
            "
            query-endpoint="/karpathy/convnetjs/refs"
            
            cache-key="v0:1388883123.0"
            current-committish="bWFzdGVy"
            default-branch="bWFzdGVy"
            name-with-owner="a2FycGF0aHkvY29udm5ldGpz"
            prefetch-on-mouseover
          >

            <template data-target="ref-selector.fetchFailedTemplate">
              <div class="SelectMenu-message" data-index="{{ index }}">Could not load branches</div>
            </template>

              <template data-target="ref-selector.noMatchTemplate">
    <div class="SelectMenu-message">Nothing to show</div>
</template>


            <div data-target="ref-selector.listContainer" role="menu" class="SelectMenu-list " data-turbo-frame="repo-content-turbo-frame">
              <div class="SelectMenu-loading pt-3 pb-0 overflow-hidden" aria-label="Menu is loading">
                <svg style="box-sizing: content-box; color: var(--color-icon-primary);" width="32" height="32" viewBox="0 0 16 16" fill="none" data-view-component="true" class="anim-rotate">
  <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-opacity="0.25" stroke-width="2" vector-effect="non-scaling-stroke" />
  <path d="M15 8a7.002 7.002 0 00-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke" />
</svg>
              </div>
            </div>

              

<template data-target="ref-selector.itemTemplate">
  <a href="https://github.com/karpathy/convnetjs/tree/{{ urlEncodedRefName }}" class="SelectMenu-item" role="menuitemradio" rel="nofollow" aria-checked="{{ isCurrent }}" data-index="{{ index }}" >
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    <span class="flex-1 css-truncate css-truncate-overflow {{ isFilteringClass }}">{{ refName }}</span>
    <span hidden="{{ isNotDefault }}" class="Label Label--secondary flex-self-start">default</span>
  </a>
</template>


              <footer class="SelectMenu-footer"><a href="/karpathy/convnetjs/branches">View all branches</a></footer>
          </ref-selector>

        </div>

        <div role="tabpanel" id="tags-menu" data-filter-placeholder="Find a tag" tabindex="" hidden class="d-flex flex-column flex-auto overflow-auto">
          <ref-selector
            type="tag"
            data-action="
              input-entered:ref-selector#inputEntered
              tab-selected:ref-selector#tabSelected
              focus-list:ref-selector#focusFirstListMember
            "
            data-targets="input-demux.sinks"
            query-endpoint="/karpathy/convnetjs/refs"
            cache-key="v0:1388883123.0"
            current-committish="bWFzdGVy"
            default-branch="bWFzdGVy"
            name-with-owner="a2FycGF0aHkvY29udm5ldGpz"
          >

            <template data-target="ref-selector.fetchFailedTemplate">
              <div class="SelectMenu-message" data-index="{{ index }}">Could not load tags</div>
            </template>

            <template data-target="ref-selector.noMatchTemplate">
              <div class="SelectMenu-message" data-index="{{ index }}">Nothing to show</div>
            </template>

              

<template data-target="ref-selector.itemTemplate">
  <a href="https://github.com/karpathy/convnetjs/tree/{{ urlEncodedRefName }}" class="SelectMenu-item" role="menuitemradio" rel="nofollow" aria-checked="{{ isCurrent }}" data-index="{{ index }}" >
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check SelectMenu-icon SelectMenu-icon--check">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    <span class="flex-1 css-truncate css-truncate-overflow {{ isFilteringClass }}">{{ refName }}</span>
    <span hidden="{{ isNotDefault }}" class="Label Label--secondary flex-self-start">default</span>
  </a>
</template>


            <div data-target="ref-selector.listContainer" role="menu" class="SelectMenu-list" data-turbo-frame="repo-content-turbo-frame">
              <div class="SelectMenu-loading pt-3 pb-0 overflow-hidden" aria-label="Menu is loading">
                <svg style="box-sizing: content-box; color: var(--color-icon-primary);" width="32" height="32" viewBox="0 0 16 16" fill="none" data-view-component="true" class="anim-rotate">
  <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-opacity="0.25" stroke-width="2" vector-effect="non-scaling-stroke" />
  <path d="M15 8a7.002 7.002 0 00-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke" />
</svg>
              </div>
            </div>
              <footer class="SelectMenu-footer"><a href="/karpathy/convnetjs/tags">View all tags</a></footer>
          </ref-selector>
        </div>
      </tab-container>
    </input-demux>
  </div>
</div>

  </details>

</div>


<div class="Overlay--hidden Overlay-backdrop--center" data-modal-dialog-overlay>
  <modal-dialog role="dialog" id="warn-tag-match-create-branch-dialog" aria-modal="true" aria-labelledby="warn-tag-match-create-branch-dialog-header" data-view-component="true" class="Overlay Overlay--width-large Overlay--height-auto Overlay--motion-scaleFade">
      <header class="Overlay-header Overlay-header--large Overlay-header--divided">
        <div class="Overlay-headerContentWrap">
          <div class="Overlay-titleWrap">
            <h1 id="warn-tag-match-create-branch-dialog-header" class="Overlay-title">Name already in use</h1>
          </div>
          <div class="Overlay-actionWrap">
            <button data-close-dialog-id="warn-tag-match-create-branch-dialog" aria-label="Close" type="button" data-view-component="true" class="close-button Overlay-closeButton"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
</svg></button>
          </div>
        </div>
      </header>
    <div class="Overlay-body ">
      
          <div data-view-component="true">      A tag already exists with the provided branch name. Many Git commands accept both tag and branch names, so creating this branch may cause unexpected behavior. Are you sure you want to create this branch?
</div>

    </div>
      <footer class="Overlay-footer Overlay-footer--alignEnd">
            <button data-close-dialog-id="warn-tag-match-create-branch-dialog" type="button" data-view-component="true" class="btn">    Cancel
</button>
            <button data-submit-dialog-id="warn-tag-match-create-branch-dialog" type="button" data-view-component="true" class="btn-danger btn">    Create
</button>
      </footer>
</modal-dialog></div>



  <div class="flex-self-center flex-self-stretch d-none d-lg-flex flex-items-center lh-condensed-ultra">
    <a data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/branches" class="ml-3 Link--primary no-underline">
          <svg text="gray" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-branch">
    <path fill-rule="evenodd" d="M11.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122V6A2.5 2.5 0 0110 8.5H6a1 1 0 00-1 1v1.128a2.251 2.251 0 11-1.5 0V5.372a2.25 2.25 0 111.5 0v1.836A2.492 2.492 0 016 7h4a1 1 0 001-1v-.628A2.25 2.25 0 019.5 3.25zM4.25 12a.75.75 0 100 1.5.75.75 0 000-1.5zM3.5 3.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0z"></path>
</svg>
          <strong>1</strong>
          <span class="color-fg-muted">branch</span>
    </a>
    <a data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/tags" class="ml-3 Link--primary no-underline">
      <svg text="gray" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-tag">
    <path fill-rule="evenodd" d="M2.5 7.775V2.75a.25.25 0 01.25-.25h5.025a.25.25 0 01.177.073l6.25 6.25a.25.25 0 010 .354l-5.025 5.025a.25.25 0 01-.354 0l-6.25-6.25a.25.25 0 01-.073-.177zm-1.5 0V2.75C1 1.784 1.784 1 2.75 1h5.025c.464 0 .91.184 1.238.513l6.25 6.25a1.75 1.75 0 010 2.474l-5.026 5.026a1.75 1.75 0 01-2.474 0l-6.25-6.25A1.75 1.75 0 011 7.775zM6 5a1 1 0 100 2 1 1 0 000-2z"></path>
</svg>
        <strong>1</strong>
        <span class="color-fg-muted">tag</span>
    </a>
  </div>

  <div class="flex-auto"></div>

  <include-fragment data-test-selector="overview-actions-fragment" src="/karpathy/convnetjs/overview_actions/master"></include-fragment>


    <span class="d-none d-md-flex ml-2">
      
<get-repo class="">
    <feature-callout class="feature-callout position-relative"
                   data-query-path="/settings/notice-dismissals/codespaces_code_tab_individuals"
                   data-feature-name="codespaces_code_tab_individuals"
  >
    
    <details class="position-relative details-overlay details-reset js-codespaces-details-container"
             data-action="toggle:get-repo#onDetailsToggle"
             
    >
        <summary data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;repository_id&quot;:15642233,&quot;target&quot;:&quot;CLONE_OR_DOWNLOAD_BUTTON&quot;,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="7832373060dbb9d67a43499621e6ecf07a30e369ee3912ff48cee1b8ebced99c" data-view-component="true" class="btn-primary btn">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-code mr-1">
    <path fill-rule="evenodd" d="M4.72 3.22a.75.75 0 011.06 1.06L2.06 8l3.72 3.72a.75.75 0 11-1.06 1.06L.47 8.53a.75.75 0 010-1.06l4.25-4.25zm6.56 0a.75.75 0 10-1.06 1.06L13.94 8l-3.72 3.72a.75.75 0 101.06 1.06l4.25-4.25a.75.75 0 000-1.06l-4.25-4.25z"></path>
</svg>

        Code<span class="dropdown-caret"></span>
</summary>      <div class="position-relative">
        <div class="dropdown-menu dropdown-menu-sw p-0" style="top:6px;width:400px;max-width: calc(100vw - 320px);">
          <div
  data-target="get-repo.modal"
  
>
  <tab-container data-view-component="true">
  <div with_panel="true" data-view-component="true" class="tabnav hx_tabnav-in-dropdown color-bg-subtle m-0">
    
    <ul role="tablist" aria-label="Choose where to access your code" data-view-component="true" class="tabnav-tabs d-flex">
        <li role="presentation" data-view-component="true" class="hx_tabnav-in-dropdown-wrapper flex-1 d-inline-flex">
  <button data-tab="local" data-action="click:get-repo#localTabSelected focusin:get-repo#localTabSelected" id="local-tab" type="button" role="tab" aria-controls="local-panel" aria-selected="true" data-view-component="true" class="tabnav-tab flex-1">
    
      <span data-view-component="true">Local</span>
    
</button></li>
        <li role="presentation" data-view-component="true" class="hx_tabnav-in-dropdown-wrapper flex-1 d-inline-flex">
  <button data-tab="cloud" data-action="click:get-repo#cloudTabSelected focusin:get-repo#cloudTabSelected" data-target="feature-callout.dismisser" id="cloud-tab" type="button" role="tab" aria-controls="cloud-panel" data-view-component="true" class="tabnav-tab flex-1">
    
      <span data-view-component="true">        <span>Codespaces</span>
          <span data-targets="feature-callout.labelees" data-test-selector="codespaces-new-label" data-view-component="true" class="Label new-label-hidden Label--success ml-1">New</span>
</span>
    
</button></li>
</ul>    
</div>    <div id="local-panel" role="tabpanel" tabindex="0" aria-labelledby="local-tab" data-view-component="true">        <ul class="list-style-none">
            <li class="Box-row p-3">
  <a class="Link--muted float-right tooltipped tooltipped-s" href="https://docs.github.com/articles/which-remote-url-should-i-use" target="_blank" aria-label="Which remote URL should I use?">
  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-question">
    <path fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13zM0 8a8 8 0 1116 0A8 8 0 010 8zm9 3a1 1 0 11-2 0 1 1 0 012 0zM6.92 6.085c.081-.16.19-.299.34-.398.145-.097.371-.187.74-.187.28 0 .553.087.738.225A.613.613 0 019 6.25c0 .177-.04.264-.077.318a.956.956 0 01-.277.245c-.076.051-.158.1-.258.161l-.007.004a7.728 7.728 0 00-.313.195 2.416 2.416 0 00-.692.661.75.75 0 001.248.832.956.956 0 01.276-.245 6.3 6.3 0 01.26-.16l.006-.004c.093-.057.204-.123.313-.195.222-.149.487-.355.692-.662.214-.32.329-.702.329-1.15 0-.76-.36-1.348-.863-1.725A2.76 2.76 0 008 4c-.631 0-1.155.16-1.572.438-.413.276-.68.638-.849.977a.75.75 0 101.342.67z"></path>
</svg>
</a>

<div class="text-bold">
  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-terminal mr-2">
    <path fill-rule="evenodd" d="M0 2.75C0 1.784.784 1 1.75 1h12.5c.966 0 1.75.784 1.75 1.75v10.5A1.75 1.75 0 0114.25 15H1.75A1.75 1.75 0 010 13.25V2.75zm1.75-.25a.25.25 0 00-.25.25v10.5c0 .138.112.25.25.25h12.5a.25.25 0 00.25-.25V2.75a.25.25 0 00-.25-.25H1.75zM7.25 8a.75.75 0 01-.22.53l-2.25 2.25a.75.75 0 11-1.06-1.06L5.44 8 3.72 6.28a.75.75 0 111.06-1.06l2.25 2.25c.141.14.22.331.22.53zm1.5 1.5a.75.75 0 000 1.5h3a.75.75 0 000-1.5h-3z"></path>
</svg>
  Clone
</div>

<tab-container>

  <div class="UnderlineNav my-2 box-shadow-none">
    <div class="UnderlineNav-body" role="tablist">
          <!-- '"` --><!-- </textarea></xmp> --></option></form><form data-remote="true" data-turbo="false" action="/users/set_protocol?protocol_type=clone" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="tj4I-vmMAXaJwcekGprY_Bmuh5G9CScexmYQ21arlP01CumXWrLSLQMXcovgJWOsNvwNGczGjvzuqeBF0fmerw" />
            <button name="protocol_selector" type="submit" role="tab" class="UnderlineNav-item" aria-selected="true" value="http" data-hydro-click="{&quot;event_type&quot;:&quot;clone_or_download.click&quot;,&quot;payload&quot;:{&quot;feature_clicked&quot;:&quot;USE_HTTPS&quot;,&quot;git_repository_type&quot;:&quot;REPOSITORY&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="b12ed31f350b56ec70acca6f8deb48e3697d070f524e3a05372286363602ce57">
              HTTPS
</button></form>          <!-- '"` --><!-- </textarea></xmp> --></option></form><form data-remote="true" data-turbo="false" action="/users/set_protocol?protocol_type=clone" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="8x9ipQrOouidAp_KWAjeNmYz_ggO-S4MAKQECpjX9IlwK4PIqfBxsxfUKuWit2VmSWF0gH82h-4oa_SUH4X-2w" />
            <button name="protocol_selector" type="submit" role="tab" class="UnderlineNav-item" value="ssh" data-hydro-click="{&quot;event_type&quot;:&quot;clone_or_download.click&quot;,&quot;payload&quot;:{&quot;feature_clicked&quot;:&quot;USE_SSH&quot;,&quot;git_repository_type&quot;:&quot;REPOSITORY&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="5bccb0a16b2960677d8f09d6c493a12108f81156d700e926323f4d1090bc9260">
              SSH
</button></form>          <!-- '"` --><!-- </textarea></xmp> --></option></form><form data-remote="true" data-turbo="false" action="/users/set_protocol?protocol_type=clone" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="2YPieMb76BM5DG-CFICcx-ewWL0CBF_etX_8zglk-p1atwMVZcU7SLPa2q3uPyeXyOLSNXPL9jydsAxQjjbwzw" />
            <button name="protocol_selector" type="submit" role="tab" class="UnderlineNav-item" value="gh_cli" data-hydro-click="{&quot;event_type&quot;:&quot;clone_or_download.click&quot;,&quot;payload&quot;:{&quot;feature_clicked&quot;:&quot;USE_GH_CLI&quot;,&quot;git_repository_type&quot;:&quot;REPOSITORY&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="80b96e17949c1379edb861babed1091153b68e142499fd0da077aa0d89aaaae9">
              GitHub CLI
</button></form>    </div>
  </div>

  <div role="tabpanel">
    <div class="input-group">
  <input type="text" class="form-control input-monospace input-sm color-bg-subtle" data-autoselect value="https://github.com/karpathy/convnetjs.git" aria-label="https://github.com/karpathy/convnetjs.git" readonly>
  <div class="input-group-button">
    <clipboard-copy value="https://github.com/karpathy/convnetjs.git" aria-label="Copy to clipboard" class="btn btn-sm js-clipboard-copy tooltipped-no-delay ClipboardButton js-clone-url-http" data-copy-feedback="Copied!" data-tooltip-direction="n" data-hydro-click="{&quot;event_type&quot;:&quot;clone_or_download.click&quot;,&quot;payload&quot;:{&quot;feature_clicked&quot;:&quot;COPY_URL&quot;,&quot;git_repository_type&quot;:&quot;REPOSITORY&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="8de8990d7b07a7928079f92230532048e3a564590359128d707249a13c5fb7a7"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon d-inline-block">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-inline-block d-sm-none">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg></clipboard-copy>
  </div>
</div>

    <p class="mt-2 mb-0 f6 color-fg-muted">
        Use Git or checkout with SVN using the web URL.
    </p>
  </div>

  <div role="tabpanel" hidden>
      <div data-view-component="true" class="f6 flash flash-warn mt-2 mb-3 p-3">
  
  
        You don't have any public SSH keys in your GitHub account.
        You can <a href="/settings/ssh/new">add a new public key</a>, or try cloning this repository via HTTPS.


  
</div>
    <div class="input-group">
  <input type="text" class="form-control input-monospace input-sm color-bg-subtle" data-autoselect value="git@github.com:karpathy/convnetjs.git" aria-label="git@github.com:karpathy/convnetjs.git" readonly>
  <div class="input-group-button">
    <clipboard-copy value="git@github.com:karpathy/convnetjs.git" aria-label="Copy to clipboard" class="btn btn-sm js-clipboard-copy tooltipped-no-delay ClipboardButton js-clone-url-ssh" data-copy-feedback="Copied!" data-tooltip-direction="n" data-hydro-click="{&quot;event_type&quot;:&quot;clone_or_download.click&quot;,&quot;payload&quot;:{&quot;feature_clicked&quot;:&quot;COPY_URL&quot;,&quot;git_repository_type&quot;:&quot;REPOSITORY&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="8de8990d7b07a7928079f92230532048e3a564590359128d707249a13c5fb7a7"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon d-inline-block">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-inline-block d-sm-none">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg></clipboard-copy>
  </div>
</div>

    <p class="mt-2 mb-0 f6 color-fg-muted">
        Use a password-protected SSH key.
    </p>
  </div>

  <div role="tabpanel" hidden>
    <div class="input-group">
  <input type="text" class="form-control input-monospace input-sm color-bg-subtle" data-autoselect value="gh repo clone karpathy/convnetjs" aria-label="gh repo clone karpathy/convnetjs" readonly>
  <div class="input-group-button">
    <clipboard-copy value="gh repo clone karpathy/convnetjs" aria-label="Copy to clipboard" class="btn btn-sm js-clipboard-copy tooltipped-no-delay ClipboardButton js-clone-url-gh-cli" data-copy-feedback="Copied!" data-tooltip-direction="n" data-hydro-click="{&quot;event_type&quot;:&quot;clone_or_download.click&quot;,&quot;payload&quot;:{&quot;feature_clicked&quot;:&quot;COPY_URL&quot;,&quot;git_repository_type&quot;:&quot;REPOSITORY&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="8de8990d7b07a7928079f92230532048e3a564590359128d707249a13c5fb7a7"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon d-inline-block">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-inline-block d-sm-none">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg></clipboard-copy>
  </div>
</div>

    <p class="mt-2 mb-0 f6 color-fg-muted">
      Work fast with our official CLI.
      <a href="https://cli.github.com" target="_blank">Learn more</a>.
    </p>
  </div>
</tab-container>

</li>
<li data-platforms="windows,mac" class="Box-row Box-row--hover-gray p-3 mt-0 rounded-0 js-remove-unless-platform">
  <a class="d-flex flex-items-center color-fg-default text-bold no-underline" data-hydro-click="{&quot;event_type&quot;:&quot;clone_or_download.click&quot;,&quot;payload&quot;:{&quot;feature_clicked&quot;:&quot;OPEN_IN_DESKTOP&quot;,&quot;git_repository_type&quot;:&quot;REPOSITORY&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="62789e7a80f5d50dfdb5cdee09a87baf5a9ca94d8f66dd1da388a22c71decffd" data-action="click:get-repo#showDownloadMessage" href="https://desktop.github.com">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-desktop-download mr-2">
    <path d="M4.927 5.427l2.896 2.896a.25.25 0 00.354 0l2.896-2.896A.25.25 0 0010.896 5H8.75V.75a.75.75 0 10-1.5 0V5H5.104a.25.25 0 00-.177.427z"></path><path d="M1.573 2.573a.25.25 0 00-.073.177v7.5a.25.25 0 00.25.25h12.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-3a.75.75 0 110-1.5h3A1.75 1.75 0 0116 2.75v7.5A1.75 1.75 0 0114.25 12h-3.727c.099 1.041.52 1.872 1.292 2.757A.75.75 0 0111.25 16h-6.5a.75.75 0 01-.565-1.243c.772-.885 1.192-1.716 1.292-2.757H1.75A1.75 1.75 0 010 10.25v-7.5A1.75 1.75 0 011.75 1h3a.75.75 0 010 1.5h-3a.25.25 0 00-.177.073zM6.982 12a5.72 5.72 0 01-.765 2.5h3.566a5.72 5.72 0 01-.765-2.5H6.982z"></path>
</svg>
    Open with GitHub Desktop
</a></li>
<li class="Box-row Box-row--hover-gray p-3 mt-0" >
  <a class="d-flex flex-items-center color-fg-default text-bold no-underline" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;clone_or_download.click&quot;,&quot;payload&quot;:{&quot;feature_clicked&quot;:&quot;DOWNLOAD_ZIP&quot;,&quot;git_repository_type&quot;:&quot;REPOSITORY&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="56da9a2dbb29a70ef4b155ec047b5d7a6e36da035815ead7afb64d1a780b8b76" data-ga-click="Repository, download zip, location:repo overview" data-open-app="link" href="/karpathy/convnetjs/archive/refs/heads/master.zip">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file-zip mr-2">
    <path fill-rule="evenodd" d="M3.5 1.75a.25.25 0 01.25-.25h3a.75.75 0 000 1.5h.5a.75.75 0 000-1.5h2.086a.25.25 0 01.177.073l2.914 2.914a.25.25 0 01.073.177v8.586a.25.25 0 01-.25.25h-.5a.75.75 0 000 1.5h.5A1.75 1.75 0 0014 13.25V4.664c0-.464-.184-.909-.513-1.237L10.573.513A1.75 1.75 0 009.336 0H3.75A1.75 1.75 0 002 1.75v11.5c0 .649.353 1.214.874 1.515a.75.75 0 10.752-1.298.25.25 0 01-.126-.217V1.75zM8.75 3a.75.75 0 000 1.5h.5a.75.75 0 000-1.5h-.5zM6 5.25a.75.75 0 01.75-.75h.5a.75.75 0 010 1.5h-.5A.75.75 0 016 5.25zm2 1.5A.75.75 0 018.75 6h.5a.75.75 0 010 1.5h-.5A.75.75 0 018 6.75zm-1.25.75a.75.75 0 000 1.5h.5a.75.75 0 000-1.5h-.5zM8 9.75A.75.75 0 018.75 9h.5a.75.75 0 010 1.5h-.5A.75.75 0 018 9.75zm-.75.75a1.75 1.75 0 00-1.75 1.75v3c0 .414.336.75.75.75h2.5a.75.75 0 00.75-.75v-3a1.75 1.75 0 00-1.75-1.75h-.5zM7 12.25a.25.25 0 01.25-.25h.5a.25.25 0 01.25.25v2.25H7v-2.25z"></path>
</svg>
    Download ZIP
</a></li>

        </ul>
</div>
    <div id="cloud-panel" role="tabpanel" tabindex="0" hidden="hidden" aria-labelledby="cloud-tab" data-view-component="true" class="cloud-panel">            <div class="js-socket-channel js-updatable-content" data-channel="eyJjIjoicmVwb3NpdG9yeV9jb2Rlc3BhY2VzOjE1NjQyMjMzOjUwNTkyNzExIiwidCI6MTY2OTY2OTQwNX0=--9f6eaad1b8c7c2ef207ca0ff2011f4e5670bf42c069efcf54a555db618936138"
  data-url="/karpathy/convnetjs/codespaces/code_menu_contents"
  data-gid="MDEwOlJlcG9zaXRvcnkxNTY0MjIzMw==">
  <ul class="list-style-none">

    <li class="Box-row p-0 mt-0" >
      <include-fragment
        data-target="get-repo.codespaceList"
        src="/codespaces?codespace%5Bref%5D=master&amp;current_branch=master&amp;event_target=REPO_PAGE&amp;repo=15642233"
        data-action="include-fragment-replace:get-repo#hideSpinner"
      >
        <div
          class="d-flex flex-items-center"
          data-target="get-repo.codespaceLoadingMenu"
        >
          <svg style="box-sizing: content-box; color: var(--color-icon-primary);" width="32" height="32" viewBox="0 0 16 16" fill="none" data-view-component="true" class="my-3 flex-1 anim-rotate">
  <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-opacity="0.25" stroke-width="2" vector-effect="non-scaling-stroke" />
  <path d="M15 8a7.002 7.002 0 00-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke" />
</svg>
        </div>
      </include-fragment>
    </li>
  </ul>
</div>

</div>
</tab-container></div>


<div class="p-3" data-targets="get-repo.platforms" data-platform="mac" hidden>
  <h4 class="lh-condensed mb-3">Launching GitHub Desktop<span class="AnimatedEllipsis"></span></h4>
  <p class="color-fg-muted">
    If nothing happens, <a href="https://desktop.github.com/">download GitHub Desktop</a> and try again.
  </p>
    <button data-action="click:get-repo#onDetailsToggle" type="button" data-view-component="true" class="btn-link">
</button>
</div>
<div class="p-3" data-targets="get-repo.platforms" data-platform="windows" hidden>
  <h4 class="lh-condensed mb-3">Launching GitHub Desktop<span class="AnimatedEllipsis"></span></h4>
  <p class="color-fg-muted">
    If nothing happens, <a href="https://desktop.github.com/">download GitHub Desktop</a> and try again.
  </p>
    <button data-action="click:get-repo#onDetailsToggle" type="button" data-view-component="true" class="btn-link">
</button>
</div>
<div class="p-3" data-targets="get-repo.platforms" data-platform="xcode" hidden>
  <h4 class="lh-condensed mb-3">Launching Xcode<span class="AnimatedEllipsis"></span></h4>
  <p class="color-fg-muted">
    If nothing happens, <a href="https://developer.apple.com/xcode/">download Xcode</a> and try again.
  </p>
    <button data-action="click:get-repo#onDetailsToggle" type="button" data-view-component="true" class="btn-link">
</button>
</div>
<div class="p-3 " data-targets="get-repo.platforms" data-target="new-codespace.loadingVscode create-button.loadingVscode" data-platform="vscode" hidden>
  <poll-include-fragment data-target="get-repo.vscodePoller new-codespace.vscodePoller create-button.vscodePoller">
    <h4 class="lh-condensed mb-3">Launching Visual Studio Code<span class="AnimatedEllipsis" data-hide-on-error></span></h4>
    <p class="color-fg-muted" data-hide-on-error>Your codespace will open once ready.</p>
    <p class="color-fg-muted" data-show-on-error hidden>There was a problem preparing your codespace, please try again.</p>
  </poll-include-fragment>
</div>


        </div>
      </div>
    </details>

      <svg style="right: -7px; top: -7px;" data-targets="feature-callout.labelees" data-test-selector="codespaces-notification-callout" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-dot-fill new-label-hidden color-fg-accent position-absolute">
    <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8z"></path>
</svg>

    <form class="d-none" data-target="feature-callout.dismissalForm" data-turbo="false" action="/settings/dismiss-notice/codespaces_code_tab_individuals" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="zIAbSIZGtokIE-q3Uc7fcWULKJ71wbQD1fSdv4olDJUsTwGEj4yxH6zOR4dP_G_jX7ZzuhFTIrjQKOJmyyvX2g" autocomplete="off" /></form>
  </feature-callout>
</get-repo>

        
    </span>
</div>




      


<div class="Box mb-3" >
  <div class="Box-header position-relative">
    <h2 class="sr-only">Latest commit</h2>
    <div class="js-details-container Details d-flex rounded-top-2 flex-items-center flex-wrap" data-issue-and-pr-hovercards-enabled>
      
  <div class="flex-shrink-0 ml-n1 mr-n1 mt-n1 mb-n1 hx_avatar_stack_commit" >
    
<div class="AvatarStack flex-self-start  " >
  <div class="AvatarStack-body" aria-label="karpathy" >
      <a class="avatar avatar-user" style="width:24px;height:24px;" data-test-selector="commits-avatar-stack-avatar-link" data-hovercard-type="user" data-hovercard-url="/users/karpathy/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/karpathy">
        <img data-test-selector="commits-avatar-stack-avatar-image" src="https://avatars.githubusercontent.com/u/241138?s=48&amp;v=4" width="24" height="24" alt="@karpathy" class=" avatar-user" />
</a>  </div>
</div>

  </div>
  <div class="flex-1 d-flex flex-items-center ml-3 min-width-0">
    <div class="css-truncate css-truncate-overflow color-fg-muted" >
          <a class="commit-author user-mention" title="View all commits by karpathy" href="/karpathy/convnetjs/commits?author=karpathy">karpathy</a>
    
  

        <span class="d-none d-sm-inline">
          <a data-pjax="true" data-test-selector="commit-tease-commit-message" title="Merge pull request #80 from Larry850806/patch-1

Update Readme.md" class="Link--primary markdown-title" href="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee">Merge pull request</a> <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="191176485" data-permission-text="Title is private" data-url="https://github.com/karpathy/convnetjs/issues/80" data-hovercard-type="pull_request" data-hovercard-url="/karpathy/convnetjs/pull/80/hovercard" href="https://github.com/karpathy/convnetjs/pull/80">#80</a> <a data-pjax="true" data-test-selector="commit-tease-commit-message" title="Merge pull request #80 from Larry850806/patch-1

Update Readme.md" class="Link--primary markdown-title" href="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee">from Larry850806/patch-1</a>
        </span>
    </div>
    <span
      class="hidden-text-expander ml-2 d-inline-block "
      
    >
      <button
        type="button"
        class="color-fg-default ellipsis-expander js-details-target"
        aria-expanded="false"
        
      >
        &hellip;
      </button>
    </span>
    <div class="d-flex flex-auto flex-justify-end ml-3 flex-items-baseline">
        <a
          class="no-wrap Link--secondary no-underline mr-2 color-fg-inherit d-none d-lg-inline"
          href="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee"
          anchor="comments"
          data-pjax="#repo-content-pjax-container"
          data-turbo-frame="repo-content-turbo-frame"
          
        >
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-comment">
    <path fill-rule="evenodd" d="M2.75 2.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h2a.75.75 0 01.75.75v2.19l2.72-2.72a.75.75 0 01.53-.22h4.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25H2.75zM1 2.75C1 1.784 1.784 1 2.75 1h10.5c.966 0 1.75.784 1.75 1.75v7.5A1.75 1.75 0 0113.25 12H9.06l-2.573 2.573A1.457 1.457 0 014 13.543V12H2.75A1.75 1.75 0 011 10.25v-7.5z"></path>
</svg>
          1
        </a>
        <include-fragment accept="text/fragment+html" src="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee/rollup?direction=sw" class="d-inline" ></include-fragment>
      <a
        href="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee"
        class="f6 Link--secondary text-mono ml-2 d-none d-lg-inline"
        data-pjax="#repo-content-pjax-container"
        data-turbo-frame="repo-content-turbo-frame"
        
      >
        4c3358a
      </a>
      <a
        href="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee"
        class="Link--secondary ml-2"
        data-pjax="#repo-content-pjax-container"
        data-turbo-frame="repo-content-turbo-frame"
        
      >
        <relative-time datetime="2016-11-25T00:57:14Z" class="no-wrap">Nov 25, 2016</relative-time>
      </a>
    </div>
  </div>
  <div class="pl-0 pl-md-5 flex-order-1 width-full Details-content--hidden">
      <div class="mt-2">
        <a data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-test-selector="commit-tease-commit-message" class="Link--primary text-bold" href="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee">Merge pull request</a> <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="191176485" data-permission-text="Title is private" data-url="https://github.com/karpathy/convnetjs/issues/80" data-hovercard-type="pull_request" data-hovercard-url="/karpathy/convnetjs/pull/80/hovercard" href="https://github.com/karpathy/convnetjs/pull/80">#80</a> <a data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-test-selector="commit-tease-commit-message" class="Link--primary text-bold" href="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee">from Larry850806/patch-1</a>
      </div>
      <pre class="mt-2 text-mono color-fg-muted text-small ws-pre-wrap">Update Readme.md</pre>
    <div class="d-flex flex-items-center">
      <code class="border d-lg-none mt-2 px-1 rounded-2">4c3358a</code>
        <a
          class="no-wrap Link--secondary no-underline ml-2 mt-2 color-fg-inherit d-lg-none"
          href="/karpathy/convnetjs/commit/4c3358a315b4d71f31a0d532eb5d1700e9e592ee"
          anchor="comments"
          data-pjax="#repo-content-pjax-container"
          data-turbo-frame="repo-content-turbo-frame"
          
        >
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-comment">
    <path fill-rule="evenodd" d="M2.75 2.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h2a.75.75 0 01.75.75v2.19l2.72-2.72a.75.75 0 01.53-.22h4.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25H2.75zM1 2.75C1 1.784 1.784 1 2.75 1h10.5c.966 0 1.75.784 1.75 1.75v7.5A1.75 1.75 0 0113.25 12H9.06l-2.573 2.573A1.457 1.457 0 014 13.543V12H2.75A1.75 1.75 0 011 10.25v-7.5z"></path>
</svg>
          1
        </a>
    </div>
  </div>
      <div class="flex-shrink-0">
        <h2 class="sr-only">Git stats</h2>
        <ul class="list-style-none d-flex">
          <li class="ml-0 ml-md-3">
            <a data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/commits/master" class="pl-3 pr-3 py-3 p-md-0 mt-n3 mb-n3 mr-n3 m-md-0 Link--primary no-underline no-wrap">
              <svg text="gray" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-history">
    <path fill-rule="evenodd" d="M1.643 3.143L.427 1.927A.25.25 0 000 2.104V5.75c0 .138.112.25.25.25h3.646a.25.25 0 00.177-.427L2.715 4.215a6.5 6.5 0 11-1.18 4.458.75.75 0 10-1.493.154 8.001 8.001 0 101.6-5.684zM7.75 4a.75.75 0 01.75.75v2.992l2.028.812a.75.75 0 01-.557 1.392l-2.5-1A.75.75 0 017 8.25v-3.5A.75.75 0 017.75 4z"></path>
</svg>
              <span class="d-none d-sm-inline">
                    <strong>89</strong>
                    <span aria-label="Commits on master" class="color-fg-muted d-none d-lg-inline">
                      commits
                    </span>
              </span>
            </a>
          </li>
        </ul>
      </div>
    </div>
  </div>
    <h2 id="files"  class="sr-only">Files</h2>
    


    <a class="d-none js-permalink-shortcut" data-hotkey="y" href="/karpathy/convnetjs/tree/4c3358a315b4d71f31a0d532eb5d1700e9e592ee">Permalink</a>

  <div data-view-component="true" class="include-fragment-error flash flash-error flash-full py-2">
  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-alert">
    <path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path>
</svg>
  
    Failed to load latest commit information.


  
</div>  <div class="js-details-container Details" data-hpc>
    <div role="grid" aria-labelledby="files" class="Details-content--hidden-not-important js-navigation-container js-active-navigation-container d-md-block">
      <div class="sr-only" role="row">
        <div role="columnheader">Type</div>
        <div role="columnheader">Name</div>
        <div role="columnheader" class="d-none d-md-block">Latest commit message</div>
        <div role="columnheader">Commit time</div>
      </div>

        <div role="row" class="Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item ">
          <div role="gridcell" class="mr-3 flex-shrink-0" style="width: 16px;">
              <svg aria-label="Directory" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file-directory-fill hx_color-icon-directory">
    <path d="M1.75 1A1.75 1.75 0 000 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0016 13.25v-8.5A1.75 1.75 0 0014.25 3H7.5a.25.25 0 01-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75z"></path>
</svg>
          </div>

          <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
            <span class="css-truncate css-truncate-target d-block width-fit"><a class="js-navigation-open Link--primary" title="build" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/tree/master/build">build</a></span>
          </div>

          <div role="gridcell" class="flex-auto min-width-0 d-none d-md-block col-5 mr-3" >
              <span class="css-truncate css-truncate-target d-block width-fit markdown-title">
                    <a data-pjax="true" title="removing compiled files from repo, these will be found in releases from now on" class="Link--secondary" href="/karpathy/convnetjs/commit/c366cbedd0fb01374f7ff12518eb94e5879d94d3">removing compiled files from repo, these will be found in releases fr…</a>
              </span>
          </div>

          <div role="gridcell" class="color-fg-muted text-right" style="width:100px;">
              <time-ago datetime="2014-08-31T23:21:44Z" data-view-component="true" class="no-wrap">Sep 1, 2014</time-ago>
          </div>

        </div>
        <div role="row" class="Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item ">
          <div role="gridcell" class="mr-3 flex-shrink-0" style="width: 16px;">
              <svg aria-label="Directory" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file-directory-fill hx_color-icon-directory">
    <path d="M1.75 1A1.75 1.75 0 000 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0016 13.25v-8.5A1.75 1.75 0 0014.25 3H7.5a.25.25 0 01-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75z"></path>
</svg>
          </div>

          <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
            <span class="css-truncate css-truncate-target d-block width-fit"><a class="js-navigation-open Link--primary" title="compile" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/tree/master/compile">compile</a></span>
          </div>

          <div role="gridcell" class="flex-auto min-width-0 d-none d-md-block col-5 mr-3" >
              <span class="css-truncate css-truncate-target d-block width-fit markdown-title">
                    <a data-pjax="true" title="fixing the build script and tweaking mnist demo tiny bit" class="Link--secondary" href="/karpathy/convnetjs/commit/a5d0fd4c9647d1ff9b2eaaf74e655d0123775f0f">fixing the build script and tweaking mnist demo tiny bit</a>
              </span>
          </div>

          <div role="gridcell" class="color-fg-muted text-right" style="width:100px;">
              <time-ago datetime="2014-08-31T23:24:29Z" data-view-component="true" class="no-wrap">Sep 1, 2014</time-ago>
          </div>

        </div>
        <div role="row" class="Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item ">
          <div role="gridcell" class="mr-3 flex-shrink-0" style="width: 16px;">
              <svg aria-label="Directory" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file-directory-fill hx_color-icon-directory">
    <path d="M1.75 1A1.75 1.75 0 000 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0016 13.25v-8.5A1.75 1.75 0 0014.25 3H7.5a.25.25 0 01-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75z"></path>
</svg>
          </div>

          <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
            <span class="css-truncate css-truncate-target d-block width-fit"><a class="js-navigation-open Link--primary" title="demo" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/tree/master/demo">demo</a></span>
          </div>

          <div role="gridcell" class="flex-auto min-width-0 d-none d-md-block col-5 mr-3" >
              <span class="css-truncate css-truncate-target d-block width-fit markdown-title">
                    <a data-pjax="true" title="updated adam solver to v8 of paper

lambda parameter was redundant and has been removed." class="Link--secondary" href="/karpathy/convnetjs/commit/08cda91824fa8e2522fa42497100881146a4437e">updated adam solver to v8 of paper</a>
              </span>
          </div>

          <div role="gridcell" class="color-fg-muted text-right" style="width:100px;">
              <time-ago datetime="2015-07-28T21:45:30Z" data-view-component="true" class="no-wrap">Jul 29, 2015</time-ago>
          </div>

        </div>
        <div role="row" class="Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item ">
          <div role="gridcell" class="mr-3 flex-shrink-0" style="width: 16px;">
              <svg aria-label="Directory" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file-directory-fill hx_color-icon-directory">
    <path d="M1.75 1A1.75 1.75 0 000 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0016 13.25v-8.5A1.75 1.75 0 0014.25 3H7.5a.25.25 0 01-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75z"></path>
</svg>
          </div>

          <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
            <span class="css-truncate css-truncate-target d-block width-fit"><a class="js-navigation-open Link--primary" title="src" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/tree/master/src">src</a></span>
          </div>

          <div role="gridcell" class="flex-auto min-width-0 d-none d-md-block col-5 mr-3" >
              <span class="css-truncate css-truncate-target d-block width-fit markdown-title">
                    <a data-pjax="true" title="of by one error fixed" class="Link--secondary" href="/karpathy/convnetjs/commit/31f2fa383347801a1e3b5ffdabf170cb61d050d4">of by one error fixed</a>
              </span>
          </div>

          <div role="gridcell" class="color-fg-muted text-right" style="width:100px;">
              <time-ago datetime="2015-08-24T15:41:34Z" data-view-component="true" class="no-wrap">Aug 24, 2015</time-ago>
          </div>

        </div>
        <div role="row" class="Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item ">
          <div role="gridcell" class="mr-3 flex-shrink-0" style="width: 16px;">
              <svg aria-label="Directory" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file-directory-fill hx_color-icon-directory">
    <path d="M1.75 1A1.75 1.75 0 000 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0016 13.25v-8.5A1.75 1.75 0 0014.25 3H7.5a.25.25 0 01-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75z"></path>
</svg>
          </div>

          <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
            <span class="css-truncate css-truncate-target d-block width-fit"><a class="js-navigation-open Link--primary" title="This path skips through empty directories" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/tree/master/test/jasmine"><span class="color-fg-muted">test/</span>jasmine</a></span>
          </div>

          <div role="gridcell" class="flex-auto min-width-0 d-none d-md-block col-5 mr-3" >
              <span class="css-truncate css-truncate-target d-block width-fit markdown-title">
                    <a data-pjax="true" title="removing legacy test. switching to jasmine" class="Link--secondary" href="/karpathy/convnetjs/commit/fcf0621d01dd298e1e320562d88030367765dee3">removing legacy test. switching to jasmine</a>
              </span>
          </div>

          <div role="gridcell" class="color-fg-muted text-right" style="width:100px;">
              <time-ago datetime="2014-09-16T18:03:41Z" data-view-component="true" class="no-wrap">Sep 16, 2014</time-ago>
          </div>

        </div>
        <div role="row" class="Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item ">
          <div role="gridcell" class="mr-3 flex-shrink-0" style="width: 16px;">
              <svg aria-label="File" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file color-fg-muted">
    <path fill-rule="evenodd" d="M3.75 1.5a.25.25 0 00-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 00.25-.25V6h-2.75A1.75 1.75 0 019 4.25V1.5H3.75zm6.75.062V4.25c0 .138.112.25.25.25h2.688a.252.252 0 00-.011-.013l-2.914-2.914a.272.272 0 00-.013-.011zM2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0113.25 16h-9.5A1.75 1.75 0 012 14.25V1.75z"></path>
</svg>
          </div>

          <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
            <span class="css-truncate css-truncate-target d-block width-fit"><a class="js-navigation-open Link--primary" title="LICENSE" data-turbo-frame="repo-content-turbo-frame" itemprop="license" href="/karpathy/convnetjs/blob/master/LICENSE">LICENSE</a></span>
          </div>

          <div role="gridcell" class="flex-auto min-width-0 d-none d-md-block col-5 mr-3" >
              <span class="css-truncate css-truncate-target d-block width-fit markdown-title">
                    <a data-pjax="true" title="Modularizing the library since project was getting too big. Also adding LocallyConnectedLayer. Also adding support for both L1 (sparsity-enducing) and L2 regularization of weights. Misc bug fixed I encountered that incorrectly ignored some options passed in by user." class="Link--secondary" href="/karpathy/convnetjs/commit/b98f5adc7accce9bfb175a94048a57b0fa339db3">Modularizing the library since project was getting too big. Also addi…</a>
              </span>
          </div>

          <div role="gridcell" class="color-fg-muted text-right" style="width:100px;">
              <time-ago datetime="2014-01-09T11:06:06Z" data-view-component="true" class="no-wrap">Jan 9, 2014</time-ago>
          </div>

        </div>
        <div role="row" class="Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item ">
          <div role="gridcell" class="mr-3 flex-shrink-0" style="width: 16px;">
              <svg aria-label="File" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file color-fg-muted">
    <path fill-rule="evenodd" d="M3.75 1.5a.25.25 0 00-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 00.25-.25V6h-2.75A1.75 1.75 0 019 4.25V1.5H3.75zm6.75.062V4.25c0 .138.112.25.25.25h2.688a.252.252 0 00-.011-.013l-2.914-2.914a.272.272 0 00-.013-.011zM2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0113.25 16h-9.5A1.75 1.75 0 012 14.25V1.75z"></path>
</svg>
          </div>

          <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
            <span class="css-truncate css-truncate-target d-block width-fit"><a class="js-navigation-open Link--primary" title="Readme.md" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/blob/master/Readme.md">Readme.md</a></span>
          </div>

          <div role="gridcell" class="flex-auto min-width-0 d-none d-md-block col-5 mr-3" >
              <span class="css-truncate css-truncate-target d-block width-fit markdown-title">
                    <a data-pjax="true" title="Update Readme.md

Fix usage of document.getElementById" class="Link--secondary" href="/karpathy/convnetjs/commit/b9db2ffa44584f99aeec918154fefdce52add7c6">Update Readme.md</a>
              </span>
          </div>

          <div role="gridcell" class="color-fg-muted text-right" style="width:100px;">
              <time-ago datetime="2016-11-23T02:38:09Z" data-view-component="true" class="no-wrap">Nov 23, 2016</time-ago>
          </div>

        </div>
        <div role="row" class="Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item ">
          <div role="gridcell" class="mr-3 flex-shrink-0" style="width: 16px;">
              <svg aria-label="File" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-file color-fg-muted">
    <path fill-rule="evenodd" d="M3.75 1.5a.25.25 0 00-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 00.25-.25V6h-2.75A1.75 1.75 0 019 4.25V1.5H3.75zm6.75.062V4.25c0 .138.112.25.25.25h2.688a.252.252 0 00-.011-.013l-2.914-2.914a.272.272 0 00-.013-.011zM2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0113.25 16h-9.5A1.75 1.75 0 012 14.25V1.75z"></path>
</svg>
          </div>

          <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
            <span class="css-truncate css-truncate-target d-block width-fit"><a class="js-navigation-open Link--primary" title="bower.json" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/blob/master/bower.json">bower.json</a></span>
          </div>

          <div role="gridcell" class="flex-auto min-width-0 d-none d-md-block col-5 mr-3" >
              <span class="css-truncate css-truncate-target d-block width-fit markdown-title">
                    <a data-pjax="true" title="Added a JSON file that allows this package to be added to the bower package manager." class="Link--secondary" href="/karpathy/convnetjs/commit/f2cb00e0e8f302418027b05278b8e88b63078fcf">Added a JSON file that allows this package to be added to the bower p…</a>
              </span>
          </div>

          <div role="gridcell" class="color-fg-muted text-right" style="width:100px;">
              <time-ago datetime="2014-06-28T19:47:59Z" data-view-component="true" class="no-wrap">Jun 28, 2014</time-ago>
          </div>

        </div>
    </div>
    <div class="Details-content--shown Box-footer d-md-none p-0">
        <button aria-expanded="false" type="button" data-view-component="true" class="js-details-target btn-link d-block width-full px-3 py-2">    View code
</button>    </div>
  </div>




</div>

  
      <readme-toc>

      <div id="readme" class="Box md js-code-block-container js-code-nav-container js-tagsearch-file Box--responsive"
          data-tagsearch-path="Readme.md"
          data-tagsearch-lang="Markdown">

        <div class="d-flex  js-sticky js-position-sticky top-0 border-top-0 border-bottom p-2 flex-items-center flex-justify-between color-bg-default rounded-top-2"  style="position: sticky; z-index: 30;" >
          <div class="d-flex flex-items-center">
              <details
  data-target="readme-toc.trigger"
  data-menu-hydro-click="{&quot;event_type&quot;:&quot;repository_toc_menu.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;trigger&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}"
  data-menu-hydro-click-hmac="876edccafc89c9abf0a6ef331739502886b51606eb3a385ad20cb1460986d8ce"
  class="dropdown details-reset details-overlay"
>
  <summary
    class="btn btn-octicon m-0 mr-2 p-2"
    aria-haspopup="true"
    aria-label="Table of Contents">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-list-unordered">
    <path fill-rule="evenodd" d="M2 4a1 1 0 100-2 1 1 0 000 2zm3.75-1.5a.75.75 0 000 1.5h8.5a.75.75 0 000-1.5h-8.5zm0 5a.75.75 0 000 1.5h8.5a.75.75 0 000-1.5h-8.5zm0 5a.75.75 0 000 1.5h8.5a.75.75 0 000-1.5h-8.5zM3 8a1 1 0 11-2 0 1 1 0 012 0zm-1 6a1 1 0 100-2 1 1 0 000 2z"></path>
</svg>
  </summary>


  <details-menu class="SelectMenu" role="menu">
    <div class="SelectMenu-modal rounded-3 mt-1" style="max-height:340px;">


      <div class="SelectMenu-list SelectMenu-list--borderless p-2" style="overscroll-behavior: contain;">
          <a role="menuitem" class="filter-item SelectMenu-item ws-normal wb-break-word line-clamp-2 py-1 text-emphasized" style="-webkit-box-orient: vertical; padding-left: 12px;" data-action="click:readme-toc#blur" data-targets="readme-toc.entries" data-hydro-click="{&quot;event_type&quot;:&quot;repository_toc_menu.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;entry&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="d3b390f8c1d48d53f6bb7fba0cbde06a15900a0fa64e523284c9089aab6a457a" href="#convnetjs">ConvNetJS</a>
          <a role="menuitem" class="filter-item SelectMenu-item ws-normal wb-break-word line-clamp-2 py-1 " style="-webkit-box-orient: vertical; padding-left: 24px;" data-action="click:readme-toc#blur" data-targets="readme-toc.entries" data-hydro-click="{&quot;event_type&quot;:&quot;repository_toc_menu.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;entry&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="d3b390f8c1d48d53f6bb7fba0cbde06a15900a0fa64e523284c9089aab6a457a" href="#online-demos">Online Demos</a>
          <a role="menuitem" class="filter-item SelectMenu-item ws-normal wb-break-word line-clamp-2 py-1 " style="-webkit-box-orient: vertical; padding-left: 24px;" data-action="click:readme-toc#blur" data-targets="readme-toc.entries" data-hydro-click="{&quot;event_type&quot;:&quot;repository_toc_menu.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;entry&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="d3b390f8c1d48d53f6bb7fba0cbde06a15900a0fa64e523284c9089aab6a457a" href="#example-code">Example Code</a>
          <a role="menuitem" class="filter-item SelectMenu-item ws-normal wb-break-word line-clamp-2 py-1 " style="-webkit-box-orient: vertical; padding-left: 24px;" data-action="click:readme-toc#blur" data-targets="readme-toc.entries" data-hydro-click="{&quot;event_type&quot;:&quot;repository_toc_menu.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;entry&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="d3b390f8c1d48d53f6bb7fba0cbde06a15900a0fa64e523284c9089aab6a457a" href="#getting-started">Getting Started</a>
          <a role="menuitem" class="filter-item SelectMenu-item ws-normal wb-break-word line-clamp-2 py-1 " style="-webkit-box-orient: vertical; padding-left: 24px;" data-action="click:readme-toc#blur" data-targets="readme-toc.entries" data-hydro-click="{&quot;event_type&quot;:&quot;repository_toc_menu.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;entry&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="d3b390f8c1d48d53f6bb7fba0cbde06a15900a0fa64e523284c9089aab6a457a" href="#compiling-the-library-from-src-to-build">Compiling the library from src/ to build/</a>
          <a role="menuitem" class="filter-item SelectMenu-item ws-normal wb-break-word line-clamp-2 py-1 " style="-webkit-box-orient: vertical; padding-left: 24px;" data-action="click:readme-toc#blur" data-targets="readme-toc.entries" data-hydro-click="{&quot;event_type&quot;:&quot;repository_toc_menu.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;entry&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="d3b390f8c1d48d53f6bb7fba0cbde06a15900a0fa64e523284c9089aab6a457a" href="#use-in-node">Use in Node</a>
          <a role="menuitem" class="filter-item SelectMenu-item ws-normal wb-break-word line-clamp-2 py-1 " style="-webkit-box-orient: vertical; padding-left: 24px;" data-action="click:readme-toc#blur" data-targets="readme-toc.entries" data-hydro-click="{&quot;event_type&quot;:&quot;repository_toc_menu.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;entry&quot;,&quot;repository_id&quot;:15642233,&quot;originating_url&quot;:&quot;https://github.com/karpathy/convnetjs&quot;,&quot;user_id&quot;:50592711}}" data-hydro-click-hmac="d3b390f8c1d48d53f6bb7fba0cbde06a15900a0fa64e523284c9089aab6a457a" href="#license">License</a>
      </div>
    </div>
  </details-menu>
</details>

            <h2 class="Box-title">
              <a href="#readme" data-view-component="true" class="Link--primary">Readme.md</a>
            </h2>
          </div>
        </div>

          <div data-target="readme-toc.content" class="Box-body px-5 pb-5">
            <article class="markdown-body entry-content container-lg" itemprop="text"><h1 dir="auto"><a id="user-content-convnetjs" class="anchor" aria-hidden="true" href="#convnetjs"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>ConvNetJS</h1>
<p dir="auto">ConvNetJS is a Javascript implementation of Neural networks, together with nice browser-based demos. It currently supports:</p>
<ul dir="auto">
<li>Common <strong>Neural Network modules</strong> (fully connected layers, non-linearities)</li>
<li>Classification (SVM/Softmax) and Regression (L2) <strong>cost functions</strong></li>
<li>Ability to specify and train <strong>Convolutional Networks</strong> that process images</li>
<li>An experimental <strong>Reinforcement Learning</strong> module, based on Deep Q Learning</li>
</ul>
<p dir="auto">For much more information, see the main page at <a href="http://convnetjs.com" rel="nofollow">convnetjs.com</a></p>
<p dir="auto"><strong>Note</strong>: I am not actively maintaining ConvNetJS anymore because I simply don't have time. I think the npm repo might not work at this point.</p>
<h2 dir="auto"><a id="user-content-online-demos" class="anchor" aria-hidden="true" href="#online-demos"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Online Demos</h2>
<ul dir="auto">
<li><a href="http://cs.stanford.edu/~karpathy/convnetjs/demo/mnist.html" rel="nofollow">Convolutional Neural Network on MNIST digits</a></li>
<li><a href="http://cs.stanford.edu/~karpathy/convnetjs/demo/cifar10.html" rel="nofollow">Convolutional Neural Network on CIFAR-10</a></li>
<li><a href="http://cs.stanford.edu/~karpathy/convnetjs/demo/classify2d.html" rel="nofollow">Toy 2D data</a></li>
<li><a href="http://cs.stanford.edu/~karpathy/convnetjs/demo/regression.html" rel="nofollow">Toy 1D regression</a></li>
<li><a href="http://cs.stanford.edu/~karpathy/convnetjs/demo/autoencoder.html" rel="nofollow">Training an Autoencoder on MNIST digits</a></li>
<li><a href="http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html" rel="nofollow">Deep Q Learning Reinforcement Learning demo</a></li>
<li><a href="http://cs.stanford.edu/~karpathy/convnetjs/demo/image_regression.html" rel="nofollow">Image Regression ("Painting")</a></li>
<li><a href="http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html" rel="nofollow">Comparison of SGD/Adagrad/Adadelta on MNIST</a></li>
</ul>
<h2 dir="auto"><a id="user-content-example-code" class="anchor" aria-hidden="true" href="#example-code"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Example Code</h2>
<p dir="auto">Here's a minimum example of defining a <strong>2-layer neural network</strong> and training
it on a single data point:</p>
<div class="highlight highlight-source-js notranslate position-relative overflow-auto" dir="auto" data-snippet-clipboard-copy-content="// species a 2-layer neural network with one hidden layer of 20 neurons
var layer_defs = [];
// input layer declares size of input. here: 2-D data
// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
// then the first two dimensions (sx, sy) will always be kept at size 1
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'}); 
// declare the linear classifier on top of the previous hidden layer
layer_defs.push({type:'softmax', num_classes:10});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// forward a random data point through the network
var x = new convnetjs.Vol([0.3, -0.5]);
var prob = net.forward(x); 

// prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
console.log('probability that x is class 0: ' + prob.w[0]); // prints 0.50101

var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
trainer.train(x, 0); // train the network, specifying that x is class zero

var prob2 = net.forward(x);
console.log('probability that x is class 0: ' + prob2.w[0]);
// now prints 0.50374, slightly higher than previous 0.50101: the networks
// weights have been adjusted by the Trainer to give a higher probability to
// the class we trained the network with (zero)"><pre><span class="pl-c">// species a 2-layer neural network with one hidden layer of 20 neurons</span>
<span class="pl-k">var</span> <span class="pl-s1">layer_defs</span> <span class="pl-c1">=</span> <span class="pl-kos">[</span><span class="pl-kos">]</span><span class="pl-kos">;</span>
<span class="pl-c">// input layer declares size of input. here: 2-D data</span>
<span class="pl-c">// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images</span>
<span class="pl-c">// then the first two dimensions (sx, sy) will always be kept at size 1</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'input'</span><span class="pl-kos">,</span> <span class="pl-c1">out_sx</span>:<span class="pl-c1">1</span><span class="pl-kos">,</span> <span class="pl-c1">out_sy</span>:<span class="pl-c1">1</span><span class="pl-kos">,</span> <span class="pl-c1">out_depth</span>:<span class="pl-c1">2</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'fc'</span><span class="pl-kos">,</span> <span class="pl-c1">num_neurons</span>:<span class="pl-c1">20</span><span class="pl-kos">,</span> <span class="pl-c1">activation</span>:<span class="pl-s">'relu'</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span> 
<span class="pl-c">// declare the linear classifier on top of the previous hidden layer</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'softmax'</span><span class="pl-kos">,</span> <span class="pl-c1">num_classes</span>:<span class="pl-c1">10</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>

<span class="pl-k">var</span> <span class="pl-s1">net</span> <span class="pl-c1">=</span> <span class="pl-k">new</span> <span class="pl-s1">convnetjs</span><span class="pl-kos">.</span><span class="pl-c1">Net</span><span class="pl-kos">(</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-s1">net</span><span class="pl-kos">.</span><span class="pl-en">makeLayers</span><span class="pl-kos">(</span><span class="pl-s1">layer_defs</span><span class="pl-kos">)</span><span class="pl-kos">;</span>

<span class="pl-c">// forward a random data point through the network</span>
<span class="pl-k">var</span> <span class="pl-s1">x</span> <span class="pl-c1">=</span> <span class="pl-k">new</span> <span class="pl-s1">convnetjs</span><span class="pl-kos">.</span><span class="pl-c1">Vol</span><span class="pl-kos">(</span><span class="pl-kos">[</span><span class="pl-c1">0.3</span><span class="pl-kos">,</span> <span class="pl-c1">-</span><span class="pl-c1">0.5</span><span class="pl-kos">]</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-k">var</span> <span class="pl-s1">prob</span> <span class="pl-c1">=</span> <span class="pl-s1">net</span><span class="pl-kos">.</span><span class="pl-en">forward</span><span class="pl-kos">(</span><span class="pl-s1">x</span><span class="pl-kos">)</span><span class="pl-kos">;</span> 

<span class="pl-c">// prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients</span>
<span class="pl-smi">console</span><span class="pl-kos">.</span><span class="pl-en">log</span><span class="pl-kos">(</span><span class="pl-s">'probability that x is class 0: '</span> <span class="pl-c1">+</span> <span class="pl-s1">prob</span><span class="pl-kos">.</span><span class="pl-c1">w</span><span class="pl-kos">[</span><span class="pl-c1">0</span><span class="pl-kos">]</span><span class="pl-kos">)</span><span class="pl-kos">;</span> <span class="pl-c">// prints 0.50101</span>

<span class="pl-k">var</span> <span class="pl-s1">trainer</span> <span class="pl-c1">=</span> <span class="pl-k">new</span> <span class="pl-s1">convnetjs</span><span class="pl-kos">.</span><span class="pl-c1">SGDTrainer</span><span class="pl-kos">(</span><span class="pl-s1">net</span><span class="pl-kos">,</span> <span class="pl-kos">{</span><span class="pl-c1">learning_rate</span>:<span class="pl-c1">0.01</span><span class="pl-kos">,</span> <span class="pl-c1">l2_decay</span>:<span class="pl-c1">0.001</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-s1">trainer</span><span class="pl-kos">.</span><span class="pl-en">train</span><span class="pl-kos">(</span><span class="pl-s1">x</span><span class="pl-kos">,</span> <span class="pl-c1">0</span><span class="pl-kos">)</span><span class="pl-kos">;</span> <span class="pl-c">// train the network, specifying that x is class zero</span>

<span class="pl-k">var</span> <span class="pl-s1">prob2</span> <span class="pl-c1">=</span> <span class="pl-s1">net</span><span class="pl-kos">.</span><span class="pl-en">forward</span><span class="pl-kos">(</span><span class="pl-s1">x</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-smi">console</span><span class="pl-kos">.</span><span class="pl-en">log</span><span class="pl-kos">(</span><span class="pl-s">'probability that x is class 0: '</span> <span class="pl-c1">+</span> <span class="pl-s1">prob2</span><span class="pl-kos">.</span><span class="pl-c1">w</span><span class="pl-kos">[</span><span class="pl-c1">0</span><span class="pl-kos">]</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// now prints 0.50374, slightly higher than previous 0.50101: the networks</span>
<span class="pl-c">// weights have been adjusted by the Trainer to give a higher probability to</span>
<span class="pl-c">// the class we trained the network with (zero)</span></pre></div>
<p dir="auto">and here is a small <strong>Convolutional Neural Network</strong> if you wish to predict on images:</p>
<div class="highlight highlight-source-js notranslate position-relative overflow-auto" dir="auto" data-snippet-clipboard-copy-content="var layer_defs = [];
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3}); // declare size of input
// output Vol is of size 32x32x3 here
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
// the layer will perform convolution with 16 kernels, each of size 5x5.
// the input will be padded with 2 pixels on all sides to make the output Vol of the same size
// output Vol will thus be 32x32x16 at this point
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 16x16x16 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 16x16x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 4x4x20 here
layer_defs.push({type:'softmax', num_classes:10});
// output Vol is of size 1x1x10 here

net = new convnetjs.Net();
net.makeLayers(layer_defs);

// helpful utility for converting images into Vols is included
var x = convnetjs.img_to_vol(document.getElementById('some_image'))
var output_probabilities_vol = net.forward(x)"><pre><span class="pl-k">var</span> <span class="pl-s1">layer_defs</span> <span class="pl-c1">=</span> <span class="pl-kos">[</span><span class="pl-kos">]</span><span class="pl-kos">;</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'input'</span><span class="pl-kos">,</span> <span class="pl-c1">out_sx</span>:<span class="pl-c1">32</span><span class="pl-kos">,</span> <span class="pl-c1">out_sy</span>:<span class="pl-c1">32</span><span class="pl-kos">,</span> <span class="pl-c1">out_depth</span>:<span class="pl-c1">3</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span> <span class="pl-c">// declare size of input</span>
<span class="pl-c">// output Vol is of size 32x32x3 here</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'conv'</span><span class="pl-kos">,</span> <span class="pl-c1">sx</span>:<span class="pl-c1">5</span><span class="pl-kos">,</span> <span class="pl-c1">filters</span>:<span class="pl-c1">16</span><span class="pl-kos">,</span> <span class="pl-c1">stride</span>:<span class="pl-c1">1</span><span class="pl-kos">,</span> <span class="pl-c1">pad</span>:<span class="pl-c1">2</span><span class="pl-kos">,</span> <span class="pl-c1">activation</span>:<span class="pl-s">'relu'</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// the layer will perform convolution with 16 kernels, each of size 5x5.</span>
<span class="pl-c">// the input will be padded with 2 pixels on all sides to make the output Vol of the same size</span>
<span class="pl-c">// output Vol will thus be 32x32x16 at this point</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'pool'</span><span class="pl-kos">,</span> <span class="pl-c1">sx</span>:<span class="pl-c1">2</span><span class="pl-kos">,</span> <span class="pl-c1">stride</span>:<span class="pl-c1">2</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// output Vol is of size 16x16x16 here</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'conv'</span><span class="pl-kos">,</span> <span class="pl-c1">sx</span>:<span class="pl-c1">5</span><span class="pl-kos">,</span> <span class="pl-c1">filters</span>:<span class="pl-c1">20</span><span class="pl-kos">,</span> <span class="pl-c1">stride</span>:<span class="pl-c1">1</span><span class="pl-kos">,</span> <span class="pl-c1">pad</span>:<span class="pl-c1">2</span><span class="pl-kos">,</span> <span class="pl-c1">activation</span>:<span class="pl-s">'relu'</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// output Vol is of size 16x16x20 here</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'pool'</span><span class="pl-kos">,</span> <span class="pl-c1">sx</span>:<span class="pl-c1">2</span><span class="pl-kos">,</span> <span class="pl-c1">stride</span>:<span class="pl-c1">2</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// output Vol is of size 8x8x20 here</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'conv'</span><span class="pl-kos">,</span> <span class="pl-c1">sx</span>:<span class="pl-c1">5</span><span class="pl-kos">,</span> <span class="pl-c1">filters</span>:<span class="pl-c1">20</span><span class="pl-kos">,</span> <span class="pl-c1">stride</span>:<span class="pl-c1">1</span><span class="pl-kos">,</span> <span class="pl-c1">pad</span>:<span class="pl-c1">2</span><span class="pl-kos">,</span> <span class="pl-c1">activation</span>:<span class="pl-s">'relu'</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// output Vol is of size 8x8x20 here</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'pool'</span><span class="pl-kos">,</span> <span class="pl-c1">sx</span>:<span class="pl-c1">2</span><span class="pl-kos">,</span> <span class="pl-c1">stride</span>:<span class="pl-c1">2</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// output Vol is of size 4x4x20 here</span>
<span class="pl-s1">layer_defs</span><span class="pl-kos">.</span><span class="pl-en">push</span><span class="pl-kos">(</span><span class="pl-kos">{</span><span class="pl-c1">type</span>:<span class="pl-s">'softmax'</span><span class="pl-kos">,</span> <span class="pl-c1">num_classes</span>:<span class="pl-c1">10</span><span class="pl-kos">}</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-c">// output Vol is of size 1x1x10 here</span>

<span class="pl-s1">net</span> <span class="pl-c1">=</span> <span class="pl-k">new</span> <span class="pl-s1">convnetjs</span><span class="pl-kos">.</span><span class="pl-c1">Net</span><span class="pl-kos">(</span><span class="pl-kos">)</span><span class="pl-kos">;</span>
<span class="pl-s1">net</span><span class="pl-kos">.</span><span class="pl-en">makeLayers</span><span class="pl-kos">(</span><span class="pl-s1">layer_defs</span><span class="pl-kos">)</span><span class="pl-kos">;</span>

<span class="pl-c">// helpful utility for converting images into Vols is included</span>
<span class="pl-k">var</span> <span class="pl-s1">x</span> <span class="pl-c1">=</span> <span class="pl-s1">convnetjs</span><span class="pl-kos">.</span><span class="pl-en">img_to_vol</span><span class="pl-kos">(</span><span class="pl-smi">document</span><span class="pl-kos">.</span><span class="pl-en">getElementById</span><span class="pl-kos">(</span><span class="pl-s">'some_image'</span><span class="pl-kos">)</span><span class="pl-kos">)</span>
<span class="pl-k">var</span> <span class="pl-s1">output_probabilities_vol</span> <span class="pl-c1">=</span> <span class="pl-s1">net</span><span class="pl-kos">.</span><span class="pl-en">forward</span><span class="pl-kos">(</span><span class="pl-s1">x</span><span class="pl-kos">)</span></pre></div>
<h2 dir="auto"><a id="user-content-getting-started" class="anchor" aria-hidden="true" href="#getting-started"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Getting Started</h2>
<p dir="auto">A <a href="http://cs.stanford.edu/people/karpathy/convnetjs/started.html" rel="nofollow">Getting Started</a> tutorial is available on main page.</p>
<p dir="auto">The full <a href="http://cs.stanford.edu/people/karpathy/convnetjs/docs.html" rel="nofollow">Documentation</a> can also be found there.</p>
<p dir="auto">See the <strong>releases</strong> page for this project to get the minified, compiled library, and a direct link to is also available below for convenience (but please host your own copy)</p>
<ul dir="auto">
<li><a href="http://cs.stanford.edu/people/karpathy/convnetjs/build/convnet.js" rel="nofollow">convnet.js</a></li>
<li><a href="http://cs.stanford.edu/people/karpathy/convnetjs/build/convnet-min.js" rel="nofollow">convnet-min.js</a></li>
</ul>
<h2 dir="auto"><a id="user-content-compiling-the-library-from-src-to-build" class="anchor" aria-hidden="true" href="#compiling-the-library-from-src-to-build"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Compiling the library from src/ to build/</h2>
<p dir="auto">If you would like to add features to the library, you will have to change the code in <code>src/</code> and then compile the library into the <code>build/</code> directory. The compilation script simply concatenates files in <code>src/</code> and then minifies the result.</p>
<p dir="auto">The compilation is done using an ant task: it compiles <code>build/convnet.js</code> by concatenating the source files in <code>src/</code> and then minifies the result into <code>build/convnet-min.js</code>. Make sure you have <strong>ant</strong> installed (on Ubuntu you can simply <em>sudo apt-get install</em> it), then cd into <code>compile/</code> directory and run:</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="$ ant -lib yuicompressor-2.4.8.jar -f build.xml"><pre class="notranslate"><code>$ ant -lib yuicompressor-2.4.8.jar -f build.xml
</code></pre></div>
<p dir="auto">The output files will be in <code>build/</code></p>
<h2 dir="auto"><a id="user-content-use-in-node" class="anchor" aria-hidden="true" href="#use-in-node"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Use in Node</h2>
<p dir="auto">The library is also available on <em>node.js</em>:</p>
<ol dir="auto">
<li>Install it: <code>$ npm install convnetjs</code></li>
<li>Use it: <code>var convnetjs = require("convnetjs");</code></li>
</ol>
<h2 dir="auto"><a id="user-content-license" class="anchor" aria-hidden="true" href="#license"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>License</h2>
<p dir="auto">MIT</p>
</article>
          </div>
      </div>

  </readme-toc>


</div>
  <div data-view-component="true" class="Layout-sidebar">      

      <div class="BorderGrid BorderGrid--spacious" data-pjax>
        <div class="BorderGrid-row hide-sm hide-md">
          <div class="BorderGrid-cell">
            <h2 class="mb-3 h4">About</h2>

    <p class="f4 my-3">
      Deep Learning in Javascript. Train Convolutional Neural Networks (or ordinary ones) in your browser.
    </p>


  <h3 class="sr-only">Resources</h3>
  <div class="mt-2">
    <a class="Link--muted" data-analytics-event="{&quot;category&quot;:&quot;Repository Overview&quot;,&quot;action&quot;:&quot;click&quot;,&quot;label&quot;:&quot;location:sidebar;file:readme&quot;}" href="#readme">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-book mr-2">
    <path fill-rule="evenodd" d="M0 1.75A.75.75 0 01.75 1h4.253c1.227 0 2.317.59 3 1.501A3.744 3.744 0 0111.006 1h4.245a.75.75 0 01.75.75v10.5a.75.75 0 01-.75.75h-4.507a2.25 2.25 0 00-1.591.659l-.622.621a.75.75 0 01-1.06 0l-.622-.621A2.25 2.25 0 005.258 13H.75a.75.75 0 01-.75-.75V1.75zm8.755 3a2.25 2.25 0 012.25-2.25H14.5v9h-3.757c-.71 0-1.4.201-1.992.572l.004-7.322zm-1.504 7.324l.004-5.073-.002-2.253A2.25 2.25 0 005.003 2.5H1.5v9h3.757a3.75 3.75 0 011.994.574z"></path>
</svg>
      Readme
</a>  </div>

<h3 class="sr-only">License</h3>
  <div class="mt-2">
    <a href="/karpathy/convnetjs/blob/master/LICENSE"
      class="Link--muted"
      
      data-analytics-event="{&quot;category&quot;:&quot;Repository Overview&quot;,&quot;action&quot;:&quot;click&quot;,&quot;label&quot;:&quot;location:sidebar;file:license&quot;}"
    >
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-law mr-2">
    <path fill-rule="evenodd" d="M8.75.75a.75.75 0 00-1.5 0V2h-.984c-.305 0-.604.08-.869.23l-1.288.737A.25.25 0 013.984 3H1.75a.75.75 0 000 1.5h.428L.066 9.192a.75.75 0 00.154.838l.53-.53-.53.53v.001l.002.002.002.002.006.006.016.015.045.04a3.514 3.514 0 00.686.45A4.492 4.492 0 003 11c.88 0 1.556-.22 2.023-.454a3.515 3.515 0 00.686-.45l.045-.04.016-.015.006-.006.002-.002.001-.002L5.25 9.5l.53.53a.75.75 0 00.154-.838L3.822 4.5h.162c.305 0 .604-.08.869-.23l1.289-.737a.25.25 0 01.124-.033h.984V13h-2.5a.75.75 0 000 1.5h6.5a.75.75 0 000-1.5h-2.5V3.5h.984a.25.25 0 01.124.033l1.29.736c.264.152.563.231.868.231h.162l-2.112 4.692a.75.75 0 00.154.838l.53-.53-.53.53v.001l.002.002.002.002.006.006.016.015.045.04a3.517 3.517 0 00.686.45A4.492 4.492 0 0013 11c.88 0 1.556-.22 2.023-.454a3.512 3.512 0 00.686-.45l.045-.04.01-.01.006-.005.006-.006.002-.002.001-.002-.529-.531.53.53a.75.75 0 00.154-.838L13.823 4.5h.427a.75.75 0 000-1.5h-2.234a.25.25 0 01-.124-.033l-1.29-.736A1.75 1.75 0 009.735 2H8.75V.75zM1.695 9.227c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L3 6.327l-1.305 2.9zm10 0c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L13 6.327l-1.305 2.9z"></path>
</svg>
     MIT license
    </a>
  </div>




<include-fragment  src="/karpathy/convnetjs/hovercards/citation/sidebar_partial?tree_name=master">
</include-fragment>

<h3 class="sr-only">Stars</h3>
<div class="mt-2">
  <a href="/karpathy/convnetjs/stargazers" data-view-component="true" class="Link--muted">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star mr-2">
    <path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25zm0 2.445L6.615 5.5a.75.75 0 01-.564.41l-3.097.45 2.24 2.184a.75.75 0 01.216.664l-.528 3.084 2.769-1.456a.75.75 0 01.698 0l2.77 1.456-.53-3.084a.75.75 0 01.216-.664l2.24-2.183-3.096-.45a.75.75 0 01-.564-.41L8 2.694v.001z"></path>
</svg>
    <strong>10.4k</strong>
    stars
</a></div>

<h3 class="sr-only">Watchers</h3>
<div class="mt-2">
  <a href="/karpathy/convnetjs/watchers" data-view-component="true" class="Link--muted">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-eye mr-2">
    <path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path>
</svg>
    <strong>601</strong>
    watching
</a></div>

<h3 class="sr-only">Forks</h3>
<div class="mt-2">
  <a href="/karpathy/convnetjs/network/members" data-view-component="true" class="Link--muted">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo-forked mr-2">
    <path fill-rule="evenodd" d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"></path>
</svg>
    <strong>2k</strong>
    forks
</a></div>

          </div>
        </div>

        
        
            <div class="BorderGrid-row">
              <div class="BorderGrid-cell">
                <h2 class="h4 mb-3" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame">
  <a href="/karpathy/convnetjs/releases" data-view-component="true" class="Link--primary no-underline">
    Releases
      <span title="1" data-view-component="true" class="Counter">1</span>
</a></h2>

  <a class="Link--primary d-flex no-underline" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" href="/karpathy/convnetjs/releases/tag/2014.08.31">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-tag flex-shrink-0 mt-1 color-fg-success">
    <path fill-rule="evenodd" d="M2.5 7.775V2.75a.25.25 0 01.25-.25h5.025a.25.25 0 01.177.073l6.25 6.25a.25.25 0 010 .354l-5.025 5.025a.25.25 0 01-.354 0l-6.25-6.25a.25.25 0 01-.073-.177zm-1.5 0V2.75C1 1.784 1.784 1 2.75 1h5.025c.464 0 .91.184 1.238.513l6.25 6.25a1.75 1.75 0 010 2.474l-5.026 5.026a1.75 1.75 0 01-2.474 0l-6.25-6.25A1.75 1.75 0 011 7.775zM6 5a1 1 0 100 2 1 1 0 000-2z"></path>
</svg>
    <div class="ml-2 min-width-0">
      <div class="d-flex">
        <span class="css-truncate css-truncate-target text-bold mr-2" style="max-width: none;">v1.1.0</span>
        <span title="Label: Latest" data-view-component="true" class="Label Label--success flex-shrink-0">
          Latest
</span>      </div>
      <div class="text-small color-fg-muted"><relative-time datetime="2014-09-01T00:23:09Z" class="no-wrap">Sep 1, 2014</relative-time></div>
    </div>
</a>
              </div>
            </div>

        
        
            <div class="BorderGrid-row">
              <div class="BorderGrid-cell">
                <h2 class="h4 mb-3">
  <a href="/users/karpathy/packages?repo_name=convnetjs" data-view-component="true" class="Link--primary no-underline">
    Packages <span title="0" hidden="hidden" data-view-component="true" class="Counter">0</span>
</a></h2>


      <div class="text-small color-fg-muted">
        No packages published <br>
      </div>



              </div>
            </div>

        
            <div class="BorderGrid-row" hidden>
              <div class="BorderGrid-cell">
                <include-fragment src="/karpathy/convnetjs/used_by_list" accept="text/fragment+html">
</include-fragment>
              </div>
            </div>

        
            <div class="BorderGrid-row">
              <div class="BorderGrid-cell">
                <h2 class="h4 mb-3">
  <a href="/karpathy/convnetjs/graphs/contributors" data-view-component="true" class="Link--primary no-underline">
    Contributors <span title="14" data-view-component="true" class="Counter">14</span>
</a></h2>


    <include-fragment src="/karpathy/convnetjs/contributors_list?count=14&amp;current_repository=convnetjs&amp;items_to_show=11" aria-busy="true" aria-label="Loading contributors">
      <ul class="list-style-none d-flex flex-wrap mb-n2">
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
          <li class="mb-2 ">
            <div class="Skeleton avatar avatar-user mr-2" style="width:32px;height:32px;"></div>
          </li>
      </ul>
</include-fragment>

  <div data-view-component="true" class="mt-3">
    <a text="small" href="/karpathy/convnetjs/graphs/contributors" data-view-component="true">
      + 3 contributors
</a></div>
              </div>
            </div>

        
        
            <div class="BorderGrid-row">
              <div class="BorderGrid-cell">
                <h2 class="h4 mb-3">Languages</h2>
<div class="mb-2">
  <span data-view-component="true" class="Progress">
    <span style="background-color:#f1e05a !important;;width: 59.8%;" itemprop="keywords" aria-label="JavaScript 59.8" data-view-component="true" class="Progress-item color-bg-success-emphasis"></span>
    <span style="background-color:#e34c26 !important;;width: 38.8%;" itemprop="keywords" aria-label="HTML 38.8" data-view-component="true" class="Progress-item color-bg-success-emphasis"></span>
    <span style="background-color:#563d7c !important;;width: 1.4%;" itemprop="keywords" aria-label="CSS 1.4" data-view-component="true" class="Progress-item color-bg-success-emphasis"></span>
</span></div>
<ul class="list-style-none">
    <li class="d-inline">
        <a class="d-inline-flex flex-items-center flex-nowrap Link--secondary no-underline text-small mr-3" href="/karpathy/convnetjs/search?l=javascript"  data-ga-click="Repository, language stats search click, location:repo overview">
          <svg style="color:#f1e05a;" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-dot-fill mr-2">
    <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8z"></path>
</svg>
          <span class="color-fg-default text-bold mr-1">JavaScript</span>
          <span>59.8%</span>
        </a>
    </li>
    <li class="d-inline">
        <a class="d-inline-flex flex-items-center flex-nowrap Link--secondary no-underline text-small mr-3" href="/karpathy/convnetjs/search?l=html"  data-ga-click="Repository, language stats search click, location:repo overview">
          <svg style="color:#e34c26;" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-dot-fill mr-2">
    <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8z"></path>
</svg>
          <span class="color-fg-default text-bold mr-1">HTML</span>
          <span>38.8%</span>
        </a>
    </li>
    <li class="d-inline">
        <a class="d-inline-flex flex-items-center flex-nowrap Link--secondary no-underline text-small mr-3" href="/karpathy/convnetjs/search?l=css"  data-ga-click="Repository, language stats search click, location:repo overview">
          <svg style="color:#563d7c;" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-dot-fill mr-2">
    <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8z"></path>
</svg>
          <span class="color-fg-default text-bold mr-1">CSS</span>
          <span>1.4%</span>
        </a>
    </li>
</ul>

              </div>
            </div>

        
      </div>
</div>
  
</div></div>

  </div>


  </div>

  </turbo-frame>


    </main>
  </div>

  </div>

          <footer class="footer width-full container-xl p-responsive">
  <h2 class='sr-only'>Footer</h2>

  <div class="position-relative d-flex flex-items-center pb-2 f6 color-fg-muted border-top color-border-muted flex-column-reverse flex-lg-row flex-wrap flex-lg-nowrap mt-6 pt-6">
    <div class="list-style-none d-flex flex-wrap col-0 col-lg-2 flex-justify-start flex-lg-justify-between mb-2 mb-lg-0">
      <div class="mt-2 mt-lg-0 d-flex flex-items-center">
        <a aria-label="Homepage" title="GitHub" class="footer-octicon mr-2" href="https://github.com">
          <svg aria-hidden="true" height="24" viewBox="0 0 16 16" version="1.1" width="24" data-view-component="true" class="octicon octicon-mark-github">
    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
</svg>
</a>        <span>
        &copy; 2022 GitHub, Inc.
        </span>
      </div>
    </div>

    <nav aria-label='footer' class="col-12 col-lg-8">
      <h3 class='sr-only' id='sr-footer-heading'>Footer navigation</h3>
      <ul class="list-style-none d-flex flex-wrap col-12 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0" aria-labelledby='sr-footer-heading'>
          <li class="mr-3 mr-lg-0"><a href="https://docs.github.com/en/github/site-policy/github-terms-of-service" data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to terms&quot;,&quot;label&quot;:&quot;text:terms&quot;}">Terms</a></li>
          <li class="mr-3 mr-lg-0"><a href="https://docs.github.com/en/github/site-policy/github-privacy-statement" data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to privacy&quot;,&quot;label&quot;:&quot;text:privacy&quot;}">Privacy</a></li>
          <li class="mr-3 mr-lg-0"><a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to security&quot;,&quot;label&quot;:&quot;text:security&quot;}" href="https://github.com/security">Security</a></li>
          <li class="mr-3 mr-lg-0"><a href="https://www.githubstatus.com/" data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to status&quot;,&quot;label&quot;:&quot;text:status&quot;}">Status</a></li>
          <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to help, text:Docs" href="https://docs.github.com">Docs</a></li>
          <li class="mr-3 mr-lg-0"><a href="https://support.github.com?tags=dotcom-footer" data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to contact&quot;,&quot;label&quot;:&quot;text:contact&quot;}">Contact GitHub</a></li>
          <li class="mr-3 mr-lg-0"><a href="https://github.com/pricing" data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to Pricing&quot;,&quot;label&quot;:&quot;text:Pricing&quot;}">Pricing</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://docs.github.com" data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to api&quot;,&quot;label&quot;:&quot;text:api&quot;}">API</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://services.github.com" data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to training&quot;,&quot;label&quot;:&quot;text:training&quot;}">Training</a></li>
          <li class="mr-3 mr-lg-0"><a href="https://github.blog" data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to blog&quot;,&quot;label&quot;:&quot;text:blog&quot;}">Blog</a></li>
          <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>
      </ul>
    </nav>
  </div>

  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 color-fg-muted"></span>
  </div>
</footer>




  <div id="ajax-error-message" class="ajax-error-message flash flash-error" hidden>
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-alert">
    <path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path>
</svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
</svg>
    </button>
    You can’t perform that action at this time.
  </div>

  <div class="js-stale-session-flash flash flash-warn flash-banner" hidden
    >
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-alert">
    <path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path>
</svg>
    <span class="js-stale-session-flash-signed-in" hidden>You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="js-stale-session-flash-signed-out" hidden>You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
    <template id="site-details-dialog">
  <details class="details-reset details-overlay details-overlay-dark lh-default color-fg-default hx_rsm" open>
    <summary role="button" aria-label="Close dialog"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast hx_rsm-dialog hx_rsm-modal">
      <button class="Box-btn-octicon m-0 btn-octicon position-absolute right-0 top-0" type="button" aria-label="Close dialog" data-close-dialog>
        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
</svg>
      </button>
      <div class="octocat-spinner my-6 js-details-dialog-spinner"></div>
    </details-dialog>
  </details>
</template>

    <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box color-shadow-large" style="width:360px;">
  </div>
</div>

    <template id="snippet-clipboard-copy-button">
  <div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div>
</template>


    <style>
      .user-mention[href$="/isackodero"] {
        color: var(--color-user-mention-fg);
        background-color: var(--color-user-mention-bg);
        border-radius: 2px;
        margin-left: -2px;
        margin-right: -2px;
        padding: 0 2px;
      }
    </style>


  </body>
</html>

