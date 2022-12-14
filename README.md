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
   
