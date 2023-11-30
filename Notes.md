# Notes
## Randomized Hough Transform
- Pick a random point from the thresholded gradient points
- Pick two more points constrained distance from the first point
-  Assume these three points lie on an ellipse 
- Find the tangent line at each point, we can use these tangents to compute the centre of the ellipse
- This is done by first finding the equation of each line, we know the angle to the x-axis of each line + a point on each line therefore:
$$ \text{Given points: }X_1,X_2,X_3 \qquad \text{Tangent Lines: }T_1,T_2,T_3 \\ \text{Intersection of }T_1 \text{ and }T_2 =T_{12} \\ \text{Midpoint of } M_1 \text{ and }M_2 =M_{12} \\ {} \\ \text{Centre Point: Intersection of }T_{12}M_{12} \text{ and }T_{23}M_{23}$$
$$\theta=\arctan(\frac{y}{x})=\arctan(m)=\theta \implies \tan(\theta) =m \\ c=y-mx$$

- Substute the 3 points into the equation below to calculate a, b and c

$$ x'=x-c_x \qquad y'=y-c_y \\ \alpha x'^2+2\beta x'y'+\gamma y'^2=1$$  

$$X'_1=X_1-C \qquad p_1=\begin{bmatrix} X'^2_{1x} & X'_{1y}X'_{1x} & X'^2_{1y}\end{bmatrix} \\ Aw=\begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix}\begin{bmatrix} a \\ b \\ c\end{bmatrix}=\begin{bmatrix} 1 \\ 1 \\ 1\end{bmatrix} \implies w_k=\sum_{i=0}^N A^{-1}_{ki}$$

- Next we can use the values for a, b and c to find the semi major and semi minor axis. We can do this be substituting the centre coordinates into the above equation and solve for y:
$$ \alpha c_x^2+2\beta c_xy+\gamma y^2=1 \\ {} \\ y=\frac{-2\beta c_x \pm \sqrt{(2\beta c_x)^2-4\gamma \alpha}}{2\gamma} \\ x=\frac{-2\beta c_y \pm \sqrt{(2\beta c_y)^2-4\alpha \gamma}}{2\alpha} \\ \text{Vertical Radius = }|c_y - y|\qquad \text{Horizontal Radius =}|c_x - x|$$
- If this is ellipse is similar to an ellipse in the accumulator increase its value by 1 otherwise add it to the accumulator
- Similarity measure: Difference ratio of semi-major and semi-major axis is > Threshold and distance between centres > Threshold
- Repeat this entire process an arbitrary number of times pick the highest scoring ellipse and remove its 3 points from the list then repeat this process for as many ellipses that we want to detect