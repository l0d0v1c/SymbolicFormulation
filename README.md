# Formulation Symbolic language

## Dependencies
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
To train an autoencoder we need a list of formulations having the same ingredients at several quantities. During the FSL initialisation process you can define a "dose". In formulation recipes, the quantity of each component is often given in units (oz, parts..). FSL use the same representation:

    formulationsymboliclanguage(formulae,granulo=5)

means for each ingredient the delta between the maximum and the minimum quantity is splitted in 5 doses. So CCCD means 3 doses of C and one of D. Minor components are only represented by one letter.

Let's try encoding a recipes book of cocktails


```python
import pandas as pd
df=pd.read_excel("cocktails.xlsx")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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


# Limits of the current version
This published version is limited to
* unordered ingredients: A development version is in progress to take into account a complete sequential manufacturing process
* The cocktail generation by autoencoder's latent space exploration has been successfully tested for cocktails but it has to be assessed in other contexts

# Licence
MIT

2021/2022 https://www.rd-mediation.com


```python

```
