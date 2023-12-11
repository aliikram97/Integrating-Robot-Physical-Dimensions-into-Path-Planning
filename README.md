**Planning Algorithms** 

**Basic Planning Algorithms:** 

2D-Binary maps are taken for the plots. The plot generated shows the path with the least cost. 

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.001.png)![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.002.png)

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.003.png)![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.004.png)

**Sampling Based methods:** 

**For PRM, what are the advantages and disadvantages of the four sampling methods in comparison to each other?**

**Uniform Sampling:** 

**ADVANTAGE:** The algorithm is simple to implement. The plot/map can be divided between uniform samples. So, there will be more nodes to find the path from start to goal position. There is a surety that there will be a solution as the whole plot/map is use to creates the notes/samples.  

**DISADVANTAGE:** More storage memory is required as it covers a lot of more node as compare to other sampling methods. The method usually fails to give results in narrow passages.  

**Random Sampling:** 

**ADVANTAGE:** The algorithm is simple to implement. It can be used to evaluate results. They require less memory with respect to uniform sampling.  

**DISADVANATGE:** The algorithm may or may not give a result depending on the distributed samples. The sampling  is  not  fixed,  so  can  vary  due  to  which  the  samples  can  increase,  leading  to  increase  in computation and complexity. 

**Gaussian Sampling:**  

**ADVANTAGE:** The algorithm shows better result than random sampling method. It is simple to implement. The algorithm can be use to evaluate the results.  

**DISADVANATGE:** The gaussian sampling is dependent on the normalized value which might affect the result or smoothness of the path.  

**Bridge Sampling:** 

**ADVANTAGE:** Bridge sampling gives better result that random and gaussian  sampling methods.  The algorithm can be used to find paths in narrow passages.   

**DISADVANATGE:** The algorithm sample only along the narrow paths and will not cover free spaces. The results depend on the sample density.  

- **For RRT, what is the main difference between RRT and RRT\*? What change does it make in terms of the efficiency of the algorithms and optimality of the search result?**

RRT\* provides a shorter path than RRT which is optimized. Cost of the node is not considered in RRT, whereas cost of node in RRT\* is updated after every interval to get the optimized path. We prune the tree in RRT\* according to the lowest cost to get the optimal solution which is not done in RRT.  

` `RRT is sub-optimal whereas RRT\* is asymptotically optimal. It is computationally efficient and asymptotically optimal. 

- **Comparing between PRM and RRT, what are the advantages and disadvantages?**

In PRM, sampling is done before finding the path whereas in the RRT the tree is generated along the path. PRM is multi query whereas RRT is single query. PRM has various techniques to sample the path between the start and the goal whereas the RRT uses a tree structure.  

**Advantage:** 

- It works in higher dimensions. 
- It works on practical (real-time) problems. It is more computational efficient.  
- No need to sample the whole map/plot.  

**Disadvantage:** 

- No optimality when exploring in RRT.  

**Results:** 

WPI 2D map is taken for the plots assigning a start and a goal point. The plot generated shows 

the path with the least cost. **Uniform Sampling**  

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.005.png)

**Random Sampling**  

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.006.png)

**Gaussian Sampling**  

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.007.png)

**Bridge Sampling**  

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.008.png)

**Advance Planning Algorithm:** 

RRT and its variants (RRT\* and informed RRT) planning algorithm are plotted on the 2D WPI map.  **RRT** 

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.009.png)

**RRT\*** 

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.010.png)

RRT\* considers cost and tends to minimize it to get the optimal path which is not done by RRT. Due to which we get different results in RRT and RRT\*. RRT does not give optimized result whereas RRT\* gives an optimized result.  

The sampling method produces different types of result depending on how the samples are distributed between the goal and the start. And different sample techniques have different methods to distribute the sample data between the goal and the start. 

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.011.png)

**Informed RRT\*:** 

Informed RRT\* follows ellipsoid hysteresis.**  

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.012.png)

![](Aspose.Words.2cfdf5d8-2acb-48f5-8f8e-81bdba19f0fd.013.png)
