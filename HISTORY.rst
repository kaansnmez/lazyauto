=======
History
=======

0.1.0 (2023-08-07)
------------------

* First release on PyPI.

0.1.4 (2023-08-07)
------------------

* Fixed the graphical output (Plotly) of outlier_dedection and pairplot functions to work on Local and Notebooks.

* Fixed Outlier_dedection Pca-2 chart output not auto_scaled.

* The output of the cat_plot function was not completely clean. Distances between subplots have been fixed.

0.2.0 (2023-08-09)
------------------

cat_plot
#########

* If the number of unique values in each features in the data is more than 6, it no longer draws ``Pie chart``. 
Instead ``stripplot`` is drawn. In this way, the crowded appearance on the pie chart has been eliminated.

*  Now this function draws ``barchart`` if the target features are categorical, ``countplot`` if they are numeric.

* Added in barplot barlabel show bar size.

* Added grid.

pairplot
#########

* Removed y_axis title on Boxplot. It already says features name on the chart.

0.2.4 (2023-08-10)
------------------

cat_plot
#########

* Made a structural change because subplots narrowed the view too much. 
In 3 features, a new figure is drawn every time. In this way, it has a perfect look.

* Fixed all errors in case the number of categorical features is only one.

* Width set to 15 inc for graphic output.
