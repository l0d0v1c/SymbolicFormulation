# Formulation Symbolic language
<img src="https://github.com/l0d0v1c/SymbolicFormulation/blob/main/image.jpg" alt="SymbolicFormulation" width="200" >

FSL is based on "Formulate" library available at https://github.com/l0d0v1c/formulate and the local unit FSL.py


```python
#!pip install "https://github.com/l0d0v1c/formulate/blob/main/dist/formulate-1.3-py3-none-any.whl?raw=true"
# uncomment the line above to install Formulate
from formulate.components import components
from FSL import formulationsymboliclanguage
```

## Purpose
FSL is a language focused on formulation description and deep learning. A formulation is a list of ingredients and quantities. FSL transforms this recipe in a string inspired by SMILES language used to represent molecules. These strings may be used for instance to train a deep auto encoder and generate new formulations from existing ones
## Encoding process
Ingredients can be either major or minor. Major components are the ones usually present in significant amount, minor ones are usually additives used to modify properties of the formulation, like colouring or viscosity agents. Major ingredients are encoded in latin alphabet and minor one is greek. To be included in FSL each FORMULATE object must embed a minor <True|False> property.
### Example
Considering Air as Oxygen/Nitrogen major ingredients and a minor water additive


```python
c=components(physical={"∆Hf":True,"rho":None,"minor":None})
c.add("Water","H2O",{'∆Hf':-285.83,"rho":1.0,'minor':True})
c.add("Nitrogen","N2",{'∆Hf':0,"rho":0.01,'minor':False})
c.add("Oxygen","O2",{'∆Hf':0,"rho":0.01,'minor':False})
c.setrates({"Water":0.01,"Oxygen":0.19,'Nitrogen':0.8})
c.mixing()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>N</th>
      <th>O</th>
      <th>H</th>
      <th>∆Hf</th>
      <th>rho</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Water</td>
      <td>0.01</td>
      <td>0.0000</td>
      <td>55.50800</td>
      <td>111.01700</td>
      <td>-15865.9700</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nitrogen</td>
      <td>0.80</td>
      <td>71.3940</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.0000</td>
      <td>0.01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Oxygen</td>
      <td>0.19</td>
      <td>0.0000</td>
      <td>62.50200</td>
      <td>0.00000</td>
      <td>0.0000</td>
      <td>0.01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Formulation</td>
      <td>1.00</td>
      <td>57.1152</td>
      <td>12.43046</td>
      <td>1.11017</td>
      <td>-158.6597</td>
      <td>Non additive</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>



We can now encode the air formulation as


```python
from IPython.display import display, HTML
f=formulationsymboliclanguage([c])
e=f.encode([c])
display(HTML(f"<span style='font-size:3em'>{e[0]}</span>"))
```


<span style='font-size:3em'>ABα</span>


The dictionary of ingredients is


```python
f.dict
```




    {'Water': 'α', 'Nitrogen': 'A', 'Oxygen': 'B'}



### Formulation list with several quantities
To train an autoencoder we need a list of formulations having the same ingredients at several quantities. During the FSL initialisation process you can define a "dose". In formulation recipes, the quantity of each component is often given in units (oz, parts..). FSL uses the same representation:

    formulationsymboliclanguage(formulae,granulo=5)

means for each ingredient the delta between the maximum and the minimum quantity is splitted in 5 doses. So CCCD means 3 doses of C and one of D. Minor components are only represented by one letter.

Let's try encoding a recipes book of cocktails


```python
import pandas as pd
df=pd.read_excel("cocktails.xlsx")
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>nom</th>
      <th>categ</th>
      <th>i1</th>
      <th>d1</th>
      <th>i2</th>
      <th>d2</th>
      <th>i3</th>
      <th>d3</th>
      <th>i4</th>
      <th>d4</th>
      <th>i5</th>
      <th>d5</th>
      <th>i6</th>
      <th>d6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Gauguin</td>
      <td>Cocktail Classics</td>
      <td>Light Rum</td>
      <td>2.0</td>
      <td>Passion Fruit Syrup</td>
      <td>1.0</td>
      <td>Lemon Juice</td>
      <td>1.00</td>
      <td>Lime Juice</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Fort Lauderdale</td>
      <td>Cocktail Classics</td>
      <td>Light Rum</td>
      <td>1.5</td>
      <td>Sweet Vermouth</td>
      <td>0.5</td>
      <td>Juice of Orange</td>
      <td>0.25</td>
      <td>Juice of a Lime</td>
      <td>0.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Apple Pie</td>
      <td>Cordials and Liqueurs</td>
      <td>Apple schnapps</td>
      <td>3.0</td>
      <td>Cinnamon schnapps</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Cuban Cocktail No. 1</td>
      <td>Cocktail Classics</td>
      <td>Juice of a Lime</td>
      <td>0.5</td>
      <td>Powdered Sugar</td>
      <td>0.5</td>
      <td>Light Rum</td>
      <td>2.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Cool Carlos</td>
      <td>Cocktail Classics</td>
      <td>Dark rum</td>
      <td>1.5</td>
      <td>Cranberry Juice</td>
      <td>2.0</td>
      <td>Pineapple Juice</td>
      <td>2.00</td>
      <td>Orange curacao</td>
      <td>1.00</td>
      <td>Sour Mix</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Now we have to transform this sheet in a list of formulations. As many ingredients are only used a few times they are not usable for a deep learning training. So we can limit the major ingredients list to the ones uses in more than 30 recipes. The rare ingredients are represented as minors


```python
from collections import Counter
ingredients=[]
for i in range(1,7):
    for j in df[f"i{i}"].tolist():
        ingredients.append(j)
ingredients=Counter(ingredients)
composant={}
for name,cnt in ingredients.items():
    if cnt>30:
        composant[name]={'minor':False}
print(f"based on {len(composant)} ingredients")
listcompo=[]
for i,j in df.iterrows():
    try:
        cp=components(physical={"minor":None})
        rates={}
        for k in range(1,7):
            if j[f"d{k}"]==j[f"d{k}"] and j[f"i{k}"]==j[f"i{k}"] : #not nan
                name=j[f"i{k}"]
                if name in composant:
                    rate=j[f"d{k}"]
                    cp.add(name,"",{'minor':False})
                    rates[name]=rate
                else:
                    cp.add(name,"",{'minor':True})
                    rates[name]=0.001
                    
        cp.setrates(rates)
        cp.mixing()
    except:
        pass
    listcompo.append(cp)
```

    based on 23 ingredients


For instance we can inpect the first cocktail


```python
listcompo[0].formulationlist
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Light Rum</td>
      <td>0.666</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Passion Fruit Syrup</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lemon Juice</td>
      <td>0.333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lime Juice</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Formulation</td>
      <td>1.000</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>



Then encode the full recipe's book


```python
cocktails=formulationsymboliclanguage(listcompo,granulo=10,verbose=False)
```

As the number of minor ingredients is limited to the length of the greek alphabet some of them are not encoded. It is possible to use longer alphabet by changing the lists 

    formulationsymboliclanguage.major=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    formulationsymboliclanguage.major=list('αβγδεζηθικλμνξοπρστυφχψω')
    
so we can now get an encoded training set. You may display unencoded ingredients by specifying verbose=True
    


```python
encoded=cocktails.encode(listcompo)
```

The first cocktail is encoded as 


```python
display(HTML(f"<div style='font-size:3em;'>Encoded recipe 0 : {encoded[0]}</div>"))

```


<div style='font-size:3em;'>Encoded recipe 0 : AAAAAAAABBBBαβ</div>


If you check what means A


```python
name={j:i for i,j in cocktails.dict.items()}['A']
print(f"Ingredient: {name}")
print(f"Minimum in recipes : {cocktails.min[name]}, maximum: {cocktails.max[name]}")
print(f"One dose of {name} is {cocktails.delta[name]}")
```

    Ingredient:  Light Rum
    Minimum in recipes : 0.04, maximum: 1.0
    One dose of  Light Rum is 0.096


Encoding is a balance between accuracy (as the quantities are encoded as a number of doses) and the number of available recipes. Having long encoded FSL strings gives a good accuracy but requires a lot of recipes to train a deep encoder. For instance, let's decode the encoded recipe


```python
display(HTML("<span style='font-size:2em;'>FSL encoded recipe is:</span>"))
display(cocktails.decode([encoded[0]])[0].formulationlist)
display(HTML("<span style='font-size:2em;'>And the original recipe was:</span>"))
display(listcompo[0].formulationlist)
```


<span style='font-size:2em;'>FSL encoded recipe is:</span>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Light Rum</td>
      <td>0.633</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lemon Juice</td>
      <td>0.365</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Passion Fruit Syrup</td>
      <td>0.001</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lime Juice</td>
      <td>0.001</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Formulation</td>
      <td>1.000</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>



<span style='font-size:2em;'>And the original recipe was:</span>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Light Rum</td>
      <td>0.666</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Passion Fruit Syrup</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lemon Juice</td>
      <td>0.333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lime Juice</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Formulation</td>
      <td>1.000</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>

# Assessment in a variational autoencoder 
Assessment of FSL language in an autoencoder. In this version, we will reload a pretrained neural network.
## Reloading the pretrained neural network


```python
#!pip install tensorflow pandas textdistance
import pickle,gzip,sys
from rdmediationvaert import AE
import pandas as pd
cocktails,encodeur=pickle.load(gzip.open("cocktails.pklz"))
dataset=[]
for m in encodeur:
    if len(m)>2:
        dataset.append(m)
print(f"{len(dataset)} formulae for training")
model=AE(name='cocktailsvae')
model.reload('cocktailsvae')
```

 

## Load a formula


```python
c=dataset[0]
print(f"FSL encoded formula : {c}")
print("Decoded formula:")
cocktails.decode([c])[0].formulationlist
```

    FSL encoded formula : AAAAAAAABBBBαβ
    Decoded formula: AAAAAAAABBBBαβ





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Light Rum</td>
      <td>0.633</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lemon Juice</td>
      <td>0.365</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Passion Fruit Syrup</td>
      <td>0.001</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lime Juice</td>
      <td>0.001</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Formulation</td>
      <td>1.000</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>



## Find it in the latent space


```python
latent=model.encode(c)
latent
```




    array([[ 1.3890834 , -0.13870159, -0.00822407, -0.00487889, -0.46605322,
            -0.79323816,  0.38904732,  0.3041486 ,  0.11699133,  0.273327  ,
            -0.09223687,  0.1689527 ,  0.15887997, -0.02809681, -0.21979149,
             1.4856585 ,  2.5984235 ,  0.10420097, -0.10993379,  0.44843948,
             0.31948787, -0.09654102,  0.31869823, -0.6928068 , -0.618227  ,
            -1.1512997 , -0.58362055,  0.09300974,  0.04692227, -0.29087883,
             0.08301675, -0.15936494]], dtype=float32)



## Rebuild it back


```python
model.decode(latent)
```

 



    'AAAAAAAABBBBαβ'



## Assess performance


```python
rebuilt=[model.decode(model.encode(formula)) for formula in dataset]

comparison=pd.DataFrame([[original,new] for original,new in zip(dataset,rebuilt)],
                       columns=["Formula","Rebuilt"])
comparison.head(20)
```





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Formula</th>
      <th>Rebuilt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAAAAAAABBBBαβ</td>
      <td>AAAAAAAABBBBαβ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAAAAAAACCDγ</td>
      <td>AAAAAAAACCDγ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAAAAAAADDEE</td>
      <td>AAAAAAAADDDE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FFFFFFFFFFFζηθι</td>
      <td>FFFFFFFFFFFζηθι</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GGGGGGHHHIIκλ</td>
      <td>GGGGGGHHHIIκλ</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AAAAAAAAAAAμν</td>
      <td>AAAAAAAAAAAον</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AAAAAAAJJJJJβξ</td>
      <td>AAAAAAAJJJJJβξ</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AAAAAAAAAAAοπ</td>
      <td>AAAAAAAAAAAοπ</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HHHHHIIIIIIρ</td>
      <td>HHHHHIIIIIIρ</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HHHHHHHHHHHστυφ</td>
      <td>HHHHHHHHHHHστυφ</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DDDDDKKKKKKχ</td>
      <td>DDDDDKKKKKKχ</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AAABBBBCCKKψ</td>
      <td>AAABBBBCKKKψ</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CCCKKKKLMω</td>
      <td>CCCKKKKLMω</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CCMMMMMNNN</td>
      <td>CCMMMMMNNN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BBBBMMMMNN</td>
      <td>BBBBMMMMNN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BBBBMMMNNNN</td>
      <td>BBBBMMMNNNN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DDDMMMMMNNN</td>
      <td>DDDMMMMMNNN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>KKKMMMMMMMM</td>
      <td>KKKMMMMMMMM</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GGGGGGIIOOO</td>
      <td>GGGGGGIIOOO</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CCCMMMNNNN</td>
      <td>CCCMMMNNNN</td>
    </tr>
  </tbody>
</table>
</div>



## Sørensen text distance



```python
from statistics import mean 
import textdistance
train=mean([textdistance.sorensen(orig,new) 
            for orig,new in zip(dataset[:663],rebuilt[:663])])
test=mean([textdistance.sorensen(orig,new) 
            for orig,new in zip(dataset[663:],rebuilt[663:])])
print(f"Sørensen similarity for training set: {train*100:.2f} %")
print(f"Sørensen similarity for test set: {test*100:.2f} %")
```

    Sørensen similarity for training set: 97.79 %
    Sørensen similarity for test set: 97.95 %


## Examples of use
### Ingredient replacement
Select a Formula


```python
c=dataset[2]
print(f"FSL encoded formula : {c}")
print("Decoded formula:")
cocktails.decode([c])[0].formulationlist
```

    FSL encoded formula : AAAAAAAADDEE
    Decoded formula: AAAAAAAADDEE





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Light Rum</td>
      <td>0.594</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Juice of a Lime</td>
      <td>0.206</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Powdered Sugar</td>
      <td>0.200</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Formulation</td>
      <td>1.000</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>



### Find an ingredient in the latent space


```python
cc="EEEEE"
cocktails.decode([cc])[0].formulationlist
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Powdered Sugar</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Formulation</td>
      <td>1.0</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>




```python
B_latent=model.encode(cc)
B_latent
```




    array([[-6.45386398e-01,  4.85044532e-02,  8.29209983e-02,
             3.41801457e-02,  7.71266997e-01,  5.36016107e-01,
             2.08929375e-01,  7.18495250e-02, -3.53245795e-01,
             1.99218929e-01,  4.12274413e-02, -8.70564654e-02,
             1.17326975e-01, -2.18493879e-01, -2.59110242e-01,
            -4.27905977e-01, -2.94935942e-01, -1.74721386e-02,
             6.90681040e-02, -2.25325441e+00, -1.64082974e-01,
            -7.02380240e-02,  4.02717918e-01,  6.12576544e-01,
            -1.44361891e-03,  1.13856137e+00,  2.85031438e-01,
             5.24719916e-02, -2.52416462e-01,  6.97316080e-02,
             2.07967505e-01, -2.75261998e-02]], dtype=float32)


### Remove the ingredient and brew a new cocktail


```python
new=model.decode(latent-B_latent)
new=''.join(sorted(new))
new
```




    'AAAAAAABBBBFαβ'




```python
cocktails.decode([new])[0].formulationlist

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Light Rum</td>
      <td>0.511</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lemon Juice</td>
      <td>0.335</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pineapple Juice</td>
      <td>0.153</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Passion Fruit Syrup</td>
      <td>0.001</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lime Juice</td>
      <td>0.001</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Formulation</td>
      <td>1.000</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>



## Create a new cocktail

### Locate a random latent space vector


```python
brandnew=model.generate()
cocktails.decode([brandnew])[0].formulationlist
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Component</th>
      <th>Rate</th>
      <th>minor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sweet Vermouth</td>
      <td>0.230</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Triple Sec</td>
      <td>0.124</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Powdered Sugar</td>
      <td>0.141</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gin</td>
      <td>0.505</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Formulation</td>
      <td>1.000</td>
      <td>Non additive</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


# Limits of the current version
This published version is limited to
* unordered ingredients: A development version is in progress to take into account a complete sequential manufacturing process
* The cocktail generation by autoencoder's latent space exploration has been successfully tested for cocktails but it has to be assessed in other contexts

## Running test

A MyBinder instance allows to run this version:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/l0d0v1c/SymbolicFormulation/HEAD?labpath=readme.ipynb)


# Licence
MIT

2021/2022 https://www.rd-mediation.com

# Cite
```
@misc{Brunet2021,
  author = {Brunet, L.E.},
  title = {Symbolic formulation: an encoder for formulations focused on deep autoencoders},
  year = {2021},
  doi ={10.17601/rdmediation.2021.2}
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/l0d0v1c/SymbolicFormulation}},
  commit = {bd34a46e2581e7e73878d5826ca272c1231df0fa}
}
```
 
```python

```
