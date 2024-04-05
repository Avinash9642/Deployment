from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib as joblib
import os

model=joblib.load('model.pkl')
scaler=joblib.load('scaler.save')

app =Flask(__name__)

IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def home():
    if request.method =='POST':
        da =request.form['disbursed_amount']
        ac = request.form['asset_cost']
        et = request.form['Employment_Type']
        af = request.form['Aadhar_flag']
        pf = request.form['PAN_flag']
        pcs = request.form['PERFORM_CNS_SCORE']
        nailsm = request.form['NEW_ACCTS_IN_LAST_SIX_MONTHS']
        dailsm = request.form['DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS']
        noi = request.form['NO_OF_INQUIRIES']
        a = request.form['age']
        aaam = request.form['Average_Acct_Age_Months']
        chlm = request.form['Credit_History_Length_Months']
        no = request.form['Number_of_0']
        lar = request.form['Loan_to_Asset_Ratio']
        noa = request.form['No_of_Accts']
        tia = request.form['Tot_Inactive_Accts']
        toa = request.form['Tot_Overdue_Accts']
        tcb = request.form['Tot_Current_Balance']
        tsa = request.form['Tot_Sanctioned_Amount']
        tda = request.form['Tot_Disbursed_Amount']
        ti = request.form['Tot_Installment']
        bdr = request.form['Bal_Disburse_Ratio']
        pt = request.form['Pri_Tenure']
        st = request.form['Sec_Tenure']
        dsr = request.form['Disburse_to_Sactioned_Ratio']
        aiar = request.form['Active_to_Inactive_Acct_Ratio']
        crl = request.form['Credit_Risk_Label']
        srl = request.form['Sub_Risk_Label']

        data = np.array([[da, ac, et, af, pf, pcs, nailsm, dailsm, noi, a, aaam, chlm, no, lar, noa, tia, toa, tcb, tsa, tda, ti, bdr, pt, st, dsr, aiar, crl, srl]])
        x = scaler.transform(data)
        print(x)
        prediction = model.predict(x)
        print(prediction)
    return render_template('index.html',prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)