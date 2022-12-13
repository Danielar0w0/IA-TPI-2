#encoding: utf8

from semantic_network import *
from bayes_net import *


class MySemNet(SemanticNetwork):
    def __init__(self):
        SemanticNetwork.__init__(self)
        # IMPLEMENT HERE (if needed)
        pass

    def source_confidence(self,user):
        
        # Default values for correct and wrong
        correct = 0
        wrong = 0
        
        # Obtain all declarations AssocOne introduced by the user
        declarations = [d for d in self.query_local(user=user) if isinstance(d.relation, AssocOne)]
        
        # For each entity and for each declaraded relation
        for declaration in declarations:
            
            e1 = declaration.relation.entity1
            e2 = declaration.relation.entity2
            relname = declaration.relation.name
            
            other_e2 =  [d.relation.entity2 for d in self.query_local(e1=e1, relname=relname)]
            distinct_other_e2 = set(other_e2)
            
            # Count number of times each distinct entity2 appears
            n_e2 = {e2: other_e2.count(e2) for e2 in distinct_other_e2}
            
            if n_e2[e2] == max(n_e2.values()):
                correct +=1
            else:
                wrong += 1
            
        # Based on the total numbers of correct/wrong declarations, calculate the confidence
        return self.conf_1(correct, wrong)

    def query_with_confidence(self,entity,assoc):
        
        # Default value
        local_confidence = {}
        
        # Compute number of ocurrences, n, for each alternative value of assoc in entity (in other words, e2)
        declarations = [d for d in self.query_local(e1=entity, relname=assoc) if isinstance(d.relation, AssocOne)]
        
        all_e2 = [d.relation.entity2 for d in declarations]
        distinct_e2 = set(all_e2)
        
        n_e2 = {e2: all_e2.count(e2) for e2 in distinct_e2}
        
        # Compute the total number of declarations of assoc in entity, T
        T = len(declarations)
        
        # Compute the confidence in each value e2
        for e2 in n_e2.keys():
            n = n_e2[e2] # Number of ocurrences, n
            local_confidence[e2] = self.conf_2(n, T)
        
        pds = [d for d in self.query_local(e1=entity) if isinstance(d.relation, (Member, Subtype))]
        
        pds_confidence = {}
        for e2 in [d.relation.entity2 for d in pds]:
            
            # Call the method recursivelly for all parent entities 
            new_pds_confidence = self.query_with_confidence(e2, assoc)
            
            # If pds_confidence is empty, merge it with local_confidence
            if not pds_confidence:
                pds_confidence = new_pds_confidence
            
            # Else, update values with new ones
            else:
                for p in new_pds_confidence.keys():
                    if p not in pds_confidence:
                        pds_confidence[p] = new_pds_confidence[p]
                    else:
                        pds_confidence[p] += new_pds_confidence[p]
            
        # Average the confidence results
        for p in pds_confidence.keys():
            pds_confidence[p] = pds_confidence[p] / len(pds)

        # If there are no inherited results, the local results should be returned
        if len(pds_confidence) == 0:
            return local_confidence
        
        # If there are no local results, the inherited results should be returned with a discount of 10% 
        if len(local_confidence) == 0:
            return {e2: pds_confidence[e2]*0.9 for e2 in pds_confidence}
        
        # Else, the final confidence values are computed with 0.9 for the local confidences and 0.1 for the inherited confidences
        
        for key in local_confidence:
            local_confidence[key] = local_confidence[key]*0.9
        
        for key in pds_confidence:
            if key not in local_confidence:
                local_confidence[key] = pds_confidence[key]*0.1
            else: 
                local_confidence[key] += pds_confidence[key]*0.1
    
        return local_confidence
    
    def conf_1(self, correct, wrong):
        return (1-0.75**correct) * 0.75**wrong

    def conf_2(self, n, T):
        return n/(2*T) + (1 - n/(2*T)) * (1 - 0.95**n) * 0.95**(T-n)
    

class MyBN(BayesNet):

    def __init__(self):
        BayesNet.__init__(self)
        self.probabilities = {}
    
    def individual_probabilities(self):
        
        # Iterate over all variables
        for var in self.dependencies:
            
            if var in self.probabilities:
                continue
            
            # Calculate the probability of each variable
            self.individual_prob(var)
        
        # Return the probabilities
        return self.probabilities
    
    # Recursive function to calculate the probability of a variable
    def individual_prob(self, var):
        
        # Initialize the probability of var to 0
        if var not in self.probabilities:
            self.probabilities[var] = 0
        
        # Obtain mothers of var
        mothers = self.mothers(var)
            
        # If the variable has no mothers, we can immediatly obtain its probability
        if not mothers:
            self.probabilities[var] = list(self.dependencies[var].values())[0]  
        
        # Else, we need to calculate the probability of var
        else:
            
            # Iterate through mothers
            for mother in mothers:
                
                # If we don't have the probability of mother yet, calculate it
                if mother not in self.probabilities or self.probabilities[mother] == 0:
                    # Call recursive function
                    self.individual_prob(mother)
            
            conjugations = self.gen_conj(list(mothers))

            # Iterate through each conjugation
            for conj in conjugations:

                # Obtain conj in dependencies
                for d in self.dependencies[var].keys():
                    if len([c for c in conj if c in d]) == len(conj):
                        
                        # Obtain the conditional probability
                        partial_prob = self.dependencies[var][d]
                        
                        # Calculate partial sum
                        for mother in mothers:
                            
                            # If B, C are independent: P(A) = P(A|B, C) * P(B, C) = P(A|B, C) * P(B) * P(C)
                            # If B, C are dependent: P(A) = P(A|B, C) * P(B, C) = P(A|B, C) * P(B|C) * P(C)
                            # However, the variables in this network are all independent, so we can always assume P(A) = P(A|B, C) * P(B, C) = P(A|B, C) * P(B) * P(C)
                            
                            #if self.independent(mothers):
                                # if (mother, True) in list(d):
                                #     partial_prob *= self.probabilities[mother]
                                    
                                # else: 
                                #     partial_prob *= (1-self.probabilities[mother])
                                
                            # else:
                            #     partial_prob = (...)
                            
                            if (mother, True) in list(d):
                                partial_prob *= self.probabilities[mother]
                                
                            else: 
                                partial_prob *= (1-self.probabilities[mother])
                            
                        
                        # Add partial sum to the probability of var
                        self.probabilities[var] += partial_prob

        return
    
    # Check if variables are independent
    def independent(self, variables):
        
        all_dependencies = [self.mothers(v) for v in variables]
        for v in variables:
            if v in all_dependencies:
                return False 
        
        return True
    
    # Obtain mothers of var
    def mothers(self, var):
        
        mothers = set()
        dependencies = self.dependencies[var].keys()
        
        # Note: d = frozenset({...})
        for d in dependencies:
            d = list(d)  # d = [(...), (...)]
            for mother in d:
                mothers.add(mother[0])
            
        return mothers
    
    # Generate conjunctions
    def gen_conj(self, variables):
            
            if len(variables) == 1:
                return [
                    [(variables[0], True)],
                    [(variables[0], False)]
                ]
                
            conj = []
            remaining = self.gen_conj(variables[1:])
            for r in remaining:
                conj.append([(variables[0], True)] + r)
                conj.append([(variables[0], False)] + r)
            
            return conj

