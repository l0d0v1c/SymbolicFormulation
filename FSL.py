from formulate.components import components
class formulationsymboliclanguage():
    def __init__(self,formulae,granulo=10,verbose=False):
        self.major=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.minor=list('αβγδεζηθικλμνξοπρστυφχψω')
        self.minorpossible=list('αβγδεζηθικλμνξοπρστυφχψω')
        self.majorpossible=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.dict={}
        self.max={}
        self.min={}
        self.delta={}
        self.granulo=granulo
        self.compos={}
        for compo in formulae:
            for name,data in compo.mixture.items():

                if name not in self.dict:
                    try:
                        self.dict[name]=self.minor.pop(0) if data['minor'] else self.major.pop(0)
                        self.max[name]=self.min[name]=compo.rates[name]
                        atoms={}
                        for i,j in data.items():
                            if i in compo.atoms:
                                atoms[i]=j
                        atoms['minor']=True
                        self.compos[name]=atoms
                    except:
                        if verbose:
                            print(f"Component limit reached for {name}, remaining major ({len(self.major)}) and minor ({len(self.minor)})")
                        
                else:
                    if compo.rates[name]<self.min[name]:
                        self.min[name]=compo.rates[name]
                    if compo.rates[name]>self.max[name]:
                        self.max[name]=compo.rates[name]
    def encode(self,formulae,verbose=False):
        listcompo=[]
        for compo in formulae:
            FSL=[]
            for name,data in compo.mixture.items():
                try:
                    rate=compo.rates[name]
                    if data['minor']:
                        FSL.append(self.dict[name])
                    else:
                        if self.min[name]==self.max[name]:
                            FSL.append(self.dict[name])
                            self.delta[name]=0
                        else:
                            delta=(self.max[name]-self.min[name])/self.granulo
                            current=rate-self.min[name]
                            n=round(current / delta)
                            self.delta[name]=delta
                            for _ in range(n+1):
                                FSL.append(self.dict[name])
                except:
                    if verbose:
                        print(f"component {name} not encodable")
                    
            listcompo.append(FSL)
            res=[]
            for v in listcompo:
                v.sort()
                res.append("".join(v))
        return res
    def decode(self,formulae):
        listcompo=[]
        rates={}
        reverse={j:i for i,j in self.dict.items()}
        for FSL in formulae:
            cp=components(physical={"minor":None})
            
            FSL=list(FSL)
            while len(FSL)>0:
                c=FSL.pop(0)

                if c in self.minorpossible:
                    name=reverse[c]
                    data=self.compos[name]
                    cp.mixture[name]=data
                    
                    rates[name]=0.001
                else:
                    name=reverse[c]
                    data=self.compos[name]
                    data['minor']=False
                    cp.mixture[name]=data
                    
                    amount=self.min[name]+self.delta[name]
                    while len(FSL)>0 and FSL[0]==c:
                        FSL.pop(0)
                        amount+=self.delta[name]
                    rates[name]=amount
                cp.setrates(rates)
                cp.mixing()
                listcompo.append(cp)
       

        return listcompo
            
                    
            
                