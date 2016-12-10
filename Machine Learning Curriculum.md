<div style="text-align:center"><img src ="http://i.imgur.com/dI2Q3hn.png" /></div>
# Machine Learning Curriculum
Machine Learning is a branch of Artificial Intelligence dedicated at making machines learn from observational data without being explicitly programmed.

This is a really big list because I also point to other people's list to ensure that most of the resources are accessible from this page without you looking anywhere else.

**NOTE**: There is no particular rank for each link. The order in which they appear does not convey any meaning and should not be treated differently.

## General Machine Learning
A machine is learning when its performance P on task T improves when it gains more experience E.
 * [Artificial Intelligence, Revealed](https://code.facebook.com/pages/1902086376686983) It's a quick introduction and it's mostly Machine Learning ideas so I include it here.
 * [How do I learn machine learning? - Quora](https://www.quora.com/How-do-I-learn-machine-learning-1)
 * [Intro to Machine Learning | Udacity](https://www.udacity.com/course/intro-to-machine-learning--ud120) hands on scikit-learn (python) programming learning on core ML concepts
 * [Machine Learning: Supervised, Unsupervised & Reinforcement | Udacity](https://www.udacity.com/course/machine-learning--ud262) the 2 instructors are hilarious
 * [Machine Learning Mastery](http://machinelearningmastery.com/start-here/)
 * [Andrew Ng's Course](https://www.coursera.org/learn/machine-learning) recommended for people who want to know the details of ML algorithms under the hood, understand enough maths to be dangerous and do coding assignments in Octave programming language
 * [ML Recipes - YouTube Playlist](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal) a really nicely designed concrete actionable content for ML introduction
 * [Machine Learning is Fun Part 1](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471) simple approach to machine learning for non-maths people
 * [Machine Learning with Python - YouTube Playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

## Reinforcement Learning
Building a machine that senses the environment and then chooses the best policy (action) to do at any given state to maximize its expected long-term scalar reward is the goal of reinforcement learning.
 * [Advanced Topics: RL 2015 (COMPM050/COMPGI13)](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html) by Dave Silver
 * [An Introduction Book by Richard S. Sutton and Andrew G. Barto](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)
 * [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
 * [Lecture 10: Reinforcement Learning - YouTube](https://www.youtube.com/watch?v=IXuHxkpO5E8)
 * [A Survey Paper](https://www.jair.org/media/301/live-301-1562-jair.pdf)
 * [Deep Reinforcement Learning: A Tutorial - OpenAI](https://gym.openai.com/docs/rl)

## Deep Learning
A set of machine learning techniques specialized at training deep artificial neural networks (DNN).
 * [Deep learning | Udacity](https://www.udacity.com/course/deep-learning--ud730) recommended for fast learner who knows some ML, this course provides high abstraction ideas of deep learning, dense details put in a short period amount of time in an intuitive way
 * [Deep Learning Resources (Papers, Online Courses, Books) - deeplearning4j.org](http://deeplearning4j.org/deeplearningpapers.html)
 * [Introduction to Deep Neural Networks - deeplearning4j.org](http://deeplearning4j.org/neuralnet-overview.html)
 * [NVIDIA Deep Learning Institute](https://developer.nvidia.com/deep-learning-courses) because GPU are efficient at training Neural Networks, NVIDIA notices this market !
 * [Deep Learning Book](http://www.deeplearningbook.org/) recommended for math geeks who want to understand the theoretical side
 * [Unsupervised Feature Learning and Deep Learning](http://ufldl.stanford.edu/wiki/index.php/Main_Page)
 * [DeepMind Publications](https://deepmind.com/publications.html)
 * [DeepLearning.TV - YouTube](https://www.youtube.com/channel/UC9OeZkIwhzfv-_Cb7fCikLQ) broad overview of deep learning, no implementation, just pure ideas
 * [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
 * [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
 * [Deep Learning Summer School, Montreal 2015](http://videolectures.net/deeplearning2015_montreal/)
 * [UFLDL Deep Learning Tutorial](http://deeplearning.stanford.edu/tutorial/)
 * [Neural networks class - YouTube Playlist](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
 * http://deeplearning.net/
 * https://developer.nvidia.com/deep-learning
 * http://neuralnetworksanddeeplearning.com/index.html a hands-on online book for deep learning maths intuition
 * https://github.com/lisa-lab/DeepLearningTutorials
 * https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-i
 * https://www.kadenze.com/courses/machine-learning-for-musicians-and-artists-iv
 * [Deep Learning Lectures by Yann LeCun](https://www.college-de-france.fr/site/en-yann-lecun/course-2015-2016.htm)
 
### Recurrent Neural Networks
DNNs that remember things. They also understand sequences that vary in length.
 * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 * http://colah.github.io/posts/2015-08-Understanding-LSTMs/
 * http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
 
## Open Source Trained Models
 * [deepdream](https://github.com/google/deepdream) inceptionism - a deep model that converts an image to make it contain animal/building
 * [Magenta: Music and Art Generation with Machine Intelligence](https://github.com/tensorflow/magenta)
 * [SyntaxNet](https://github.com/tensorflow/models/tree/master/syntaxnet) (Parsey McParseface)
 * [Neural Storyteller](https://github.com/ryankiros/neural-storyteller) convert image caption into a romantic one
 * https://github.com/facebookresearch/deepmask sharp object segmentation on image at pixel-level
 * https://github.com/facebookresearch/multipathnet convnet for classifying DeepMask+SharpMask model above
 * https://github.com/tensorflow/models
 
## Interesting Techniques & Applications
 * https://deepart.io/ transfer image style to other image
 * http://www.somatic.io/
 * [WaveNet: A Generative Model for Raw Audio by DeepMind](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

## Nice Blogs & Vlogs to Follow
 * http://colah.github.io/ this guy knows how to explain
 * https://karpathy.github.io/ this guy has taught some courses on Deep Nets
 * http://www.wildml.com/ this is like a repository on ML
 * https://adeshpande3.github.io/adeshpande3.github.io/
 * http://culurciello.github.io/
 * [Sirajology's YouTube Playlists](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A/playlists) lots of dense short hilarious introduction to ML
 * [Two Minute Papers on Deep Learning Playlist](https://www.youtube.com/playlist?list=PLujxSBD-JXglGL3ERdDOhthD3jTlfudC2)
 * http://www.leviathan.ai/
 * [Welch Labs](https://www.youtube.com/user/Taylorns34/)

## Libraries and Frameworks
Taking a glance at their github statistics can give you a sense of how active/popular each library is.
 * [scikit-learn (Python)](https://github.com/scikit-learn/scikit-learn) general machine learning library, high level abstraction, geared towards beginners
 * [TensorFlow (Python)](https://github.com/tensorflow/tensorflow); [Learning TensorFlow](http://learningtensorflow.com/index.html); [Installing on Windows](https://github.com/tensorflow/tensorflow/issues/42#issuecomment-263645160); [Fresh Install on Ubuntu 16.04](https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0-rc/); [Serving](https://tensorflow.github.io/serving/); computation graph framework built by Google, has nice visualization board, probably the most popular framework nowadays
 * [Theano (Python)](https://github.com/Theano/Theano) another popular deep learning framework
 * [Caffe (Python)](https://github.com/BVLC/caffe) does best at computer vision problems
 * [Computational Network Toolkit (CNTK)](https://github.com/Microsoft/CNTK) Microsoft's framework
 * [Torch (LuaJIT)](https://github.com/torch/torch7) the most popular deep learning framework for LuaJIT
 * [DeepLearning4j (Java)](https://github.com/deeplearning4j/deeplearning4j) not so popular, preferable for you if you like Java
 * [Software Tools for Reinforcement Learning, Artificial Neural Networks and Robotics (Matlab and Python)](http://jamh-web.appspot.com/download.htm)

## Cutting-Edge Research
Steal the most recent techniques introduced by smart computer scientists (could be you).
 * https://research.facebook.com/ai/
 * http://research.google.com/pubs/MachineIntelligence.html
 * https://www.openai.com/
 * https://www.openai.com/requests-for-research/
 * [State of the art performance on each ML task](http://rodrigob.github.io/are_we_there_yet/build/)
 * http://www.gitxiv.com/
 

## Practitioner Community
 * https://www.kaggle.com
 * https://gym.openai.com
 * https://universe.openai.com/

## Thoughtful Insights for Future Research
 * [Pedro Domingos: "The Master Algorithm" | Talks at Google](https://www.youtube.com/watch?v=B8J4uefCQMc)
 * [The AI Revolution: The Road to Superintelligence](http://waitbutwhy.com/2015/01/artificial-intelligence-revolution-1.html)
 * https://ai100.stanford.edu/2016-report
 * [Why does Deep Learning work so well? - The Extraordinary Link Between Deep Neural Networks and the Nature of the Universe](https://www.technologyreview.com/s/602344/the-extraordinary-link-between-deep-neural-networks-and-the-nature-of-the-universe/)
 
## Uncategorized
 * [Artificial Intelligence: A Modern Approach (Online Book)](http://aima.cs.berkeley.edu/)
 * [The Principles of Modern Game AI](https://courses.nucl.ai/)
 * [Scipy Lecture Notes](http://www.scipy-lectures.org/index.html)
 * https://www.youtube.com/user/aicourses
 
## Other Big Lists
 * https://github.com/josephmisiti/awesome-machine-learning
 * https://github.com/ujjwalkarn/Machine-Learning-Tutorials
 * https://github.com/terryum/awesome-deep-learning-papers
 * https://github.com/ChristosChristofidis/awesome-deep-learning

## How to contribute to this list
 1. Fork this gist and modify your forked version. Make sure that your forked version is the latest because this gist is updated very frequently.
 2. Comment on this gist that you want to get merged and I'll pull your fork into this one.
 3. Or just say how would you want this list to be modified and I'll do it if it's promising.