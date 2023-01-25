from train_gcn_test import AU_set_lst

#10 sets of AU_data: 0,1,1,1,....
0,1,1,1,....
0,1,1,1,....
.....

total_count=10
AU1_count=6
AU2_count=4
AU_joint=2

def adj_matrix(AU_inc1,AU_inc2,feature):
    cnt1=0
    cnt2=0
    cnt_joint=0

    
    

    for i in range(len(feature)):
        if  feature[i][AU_inc1]=="1.0":
            cnt1+=1
        if  feature[i][AU_inc2]=="1.0":
            cnt2+=1
        if  feature[i][AU_inc2]=="1.0" and feature[i][AU_inc1]=='1.0':
            cnt_joint+=1
    

    AU1_AU2_joint_count = cnt_joint # Number of instances where both AU1 and AU4 are present
    AU2_count = cnt2 # Number of instances where AU4 is present
    total_count = len(feature) # Total number of instances in the dataset

    P_AU1_AU2 = AU1_AU2_joint_count / total_count
    P_AU2 = AU2_count / total_count
    #P_AU1_given_AU2 = P_AU1_AU2 / P_AU2
    P12=AU1_AU2_joint_count/AU2_count
    return P12
def get_all():
    adj_matrix(1,2,AU_set_lst)
    adj_matrix(2,1,AU_set_lst)
    adj_matrix(1,4,AU_set_lst)
    adj_matrix(4,1,AU_set_lst)
    adj_matrix(2,4,AU_set_lst)
    adj_matrix(4,2,AU_set_lst)

    adj_matrix(10,12,AU_set_lst)
    adj_matrix(12,10,AU_set_lst)
    adj_matrix(12,14,AU_set_lst)
    adj_matrix(14,12,AU_set_lst)
    adj_matrix(10,14,AU_set_lst)
    adj_matrix(14,10,AU_set_lst)
# ........


#1,2,4,10,12,14,15,17,25

    