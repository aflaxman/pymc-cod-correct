import pandas

Oceania = 'FJI,FSM,GUM,NCL,PNG,PYF,SLB,TON,VUT,WSM'.split(',')

X_sum = pandas.DataFrame()
for iso3 in Oceania:
    print iso3
    X = pandas.read_csv('/home/j/Project/Causes of Death/codem/estimates/corrected_estimates/draws/coef_var/M/%s_B07_subcauses_draws.csv'%iso3, index_col=None)
    X.index = zip(X['age'], X['year'], X['cause'])  # TODO: make this a multiindex
    X['env'] = 10.  # FIXME: this should be in .csv file already
    for i in range(1000):
        X['draw%d'%i] = X['draw%d'%i]*X['env']
        
    X_sum = X_sum.add(X.filter(regex='draw'), fill_value=0.)
    
# X_sum will have sums for all B07 in region
# TODO: get multiindex working, then say
# X_sum = X_sum.delevel()
X_sum.to_csv('Oceania_draws.csv')
