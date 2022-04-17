import numpy as np
from flask import Flask, request, jsonify, render_template, flash, url_for
import pickle

from werkzeug.utils import redirect
from wtforms import IntegerField, SubmitField, FloatField
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, NumberRange

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))
app.config['SECRET_KEY'] = '535f7c1feb60b39072021db75defa81a'

@app.route('/', methods=['GET', 'POST'])
def home():
    form = predictForm()
    if form.validate_on_submit():
        int_features =[]
        int_features.append(float(form.age.data))
        int_features.append(float(form.edu.data))
        int_features.append(float(form.years_emp.data))
        int_features.append(float(form.income.data))
        int_features.append(float(form.card_debt.data))
        int_features.append(float(form.other_debt.data))
        int_features.append(float(form.defaulted.data))
        int_features.append(float(form.debt_income_ratio.data))
        final_features = [np.array(int_features)]
        print(final_features)
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        flash('Customer class is {}'.format(output), 'success')
        return redirect(url_for('home'))
    return render_template('index.html', form=form)


'''@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Customer Class is {}'.format(output))
'''


class predictForm(FlaskForm):
    age = FloatField("Age", validators=[DataRequired()],render_kw={"placeholder": "Age"})
    edu = FloatField('Education', validators=[DataRequired(), NumberRange(1, 5)],render_kw={"placeholder": "Edu Ex- 1,2,3,4,5"})
    years_emp = FloatField('Years Employed', validators=[DataRequired(), NumberRange(1, 33)],render_kw={"placeholder": "Years Employed Ex-1 to 33"})
    income = FloatField('Income', validators=[DataRequired()],render_kw={"placeholder": "Income Ex- IN hundreds"})
    card_debt = FloatField('Card Debt', validators=[DataRequired()],render_kw={"placeholder": "Card Debt Ex- IN tens"})
    other_debt = FloatField('Other Debt', validators=[DataRequired()],render_kw={"placeholder": "Other Debt Ex- IN tens"})
    defaulted = FloatField('Defaulted', validators=[DataRequired(), NumberRange(-1, 1)],render_kw={"placeholder": "Defaulted Ex- 0 , 1"})
    debt_income_ratio = FloatField('Debt Income Ratio', validators=[DataRequired()],render_kw={"placeholder": "DebtIncomeRatio Ex- IN tens"})
    submit = SubmitField('Submit')


if __name__ == "__main__":
    app.run(debug=True)
