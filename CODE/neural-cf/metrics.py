import math
import pandas as pd


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # implicit: full ranked list
        self._explicit_subjects = None # explicit: user/item/label/pred
    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list
                - Implicit: [test_users, test_items, test_scores, negative_users, negative_items, negative_scores]
                - Explicit: [test_users, test_items, true_ratings]
        """
        assert isinstance(subjects, list)
        if len(subjects) == 6:  # Implicit
            test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
            neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]

            print(f"Length of test_users: {len(test_users)}")
            print(f"Length of test_items: {len(test_items)}")
            print(f"Length of test_preds: {len(test_scores)}")
            # the golden set
            test = pd.DataFrame({'user': test_users,
                                'test_item': test_items,
                                'test_score': test_scores})
            # the full set
            full = pd.DataFrame({'user': neg_users + test_users,
                                'item': neg_items + test_items,
                                'score': neg_scores + test_scores})
            full = pd.merge(full, test, on=['user'], how='left')
            # rank the items according to the scores for each user
            full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
            full.sort_values(['user', 'rank'], inplace=True)
            
            self._subjects = full
            self._explicit_subjects = None
        
        elif len(subjects) == 3: # Explicit 
            self._explicit_subjects = pd.DataFrame({
            'user': subjects[0],
            'item': subjects[1],
            'label': subjects[2]
        })
            self._explicit_subjects['pred'] = None  
            self._subjects = None  
        else:
            raise ValueError("Invalid number of elements in subjects")

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']].copy()
        test_in_top_k.loc[:, 'ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()

    def cal_rmse(self):
        """Root Mean Squared Error for explicit feedback"""
        df = self._explicit_subjects
        if df is None or 'pred' not in df.columns:
            raise ValueError("Explicit predictions not set for RMSE calculation")
        return math.sqrt(((df['label'] - df['pred']) ** 2).mean())

    def cal_mae(self):
        """Mean Absolute Error for explicit feedback"""
        df = self._explicit_subjects
        if df is None or 'pred' not in df.columns:
            raise ValueError("Explicit predictions not set for MAE calculation")
        return (df['label'] - df['pred']).abs().mean()

    def set_predictions(self, preds):
        """Set predictions for explicit feedback case"""
        if self._explicit_subjects is None:
            raise ValueError("Explicit subject data must be set before assigning predictions")
        self._explicit_subjects['pred'] = preds

    def set_explicit_subjects(self, users, items, ratings):
        """Set explicit subject data directly."""
        self._explicit_subjects = pd.DataFrame({
            'user': users,
            'item': items,
            'label': ratings
        })
        self._explicit_subjects['pred'] = None
        self._subjects = None