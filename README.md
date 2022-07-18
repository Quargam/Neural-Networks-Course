# Homework #1
#### Kapylov Maksim
---
Суть ТЗ находится в файле `technical specification.txt`.

Давайте выпишим формулы для forward propagation и  backward propagation.





## forward propagation
$$ z^{(i)} = w^{(i)}x^{(i)} +b^{(i)} $$ -- Математические операции в нейроне.

 $$ \hat{y}^{(i)} = sigmoid(z^{(i)}) $$ --  Функция активации.


 $$ \hat{L}^{(i)}  = (y^{(i)} - \hat{y}^{(i)})^2 $$ -- Функция потерь.

## backward propagation

 $$ J =  \dfrac{1}{n} \sum{  (y^{(i)} - \hat{y}^{(i)})^2 } $$

 $$ \dfrac{\partial J}{\partial w^{(i)}} =  \dfrac{\partial J}{\partial\hat{y}^{(i)}} \times \dfrac{\partial\hat{y}^{(i)}}{\partial z^{(i)}}\times \dfrac{\partial z^{(i)}}{\partial w^{(i)}} $$

 $$  \dfrac{\partial J}{\partial\hat{y}^{(i)}} = 2\times (y^{(i)} - \hat{y}^{(i)}) $$

 $$ \dfrac{\partial\hat{y}^{(i)}}{\partial z^{(i)}} =sigmoid(z^{(i)}) \times(1 -sigmoid(z^{(i)}))  $$

 $$ \dfrac{\partial z^{(i)}}{\partial w^{(i)}} = x^{(i)} $$

 $$ \dfrac{\partial J}{\partial b^{(i)}} =  \dfrac{\partial J}{\partial\hat{y}^{(i)}} \times \dfrac{\partial\hat{y}^{(i)}}{\partial z^{(i)}}\times \dfrac{\partial z^{(i)}}{\partial b^{(i)}} $$

## Обновление параметров

 $$ w = w - \alpha \dfrac{\partial J}{\partial w} $$

 $$ b = b - \alpha \dfrac{\partial J}{\partial b} $$