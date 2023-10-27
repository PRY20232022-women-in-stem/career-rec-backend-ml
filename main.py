from flask import Flask, request, jsonify
import joblib 

app = Flask(__name__)

model= joblib.load('DecisionTreeML.joblib')

@app.route("/predict",methods=['POST'])
def predict_area():
    try:
        data=request.get_json()
        input_data = [[
                        data["mathLogicAbility"], 
                        data["mathDifficulty"],
                        data["mathInterest"],
                        data["mathChallenges"],
                        data["mathExamCommitment"],
                        data["mathPerformance"],
                        data["mathParticipation"],
                        data["mathRealWorldApplication"],
                        data["mathCareerFuture"],
                        data["mathCareerImportance"],
                        data["scienceInterest"],
                        data["scienceCareerPossibility"],
                        data["scienceParticipation"],
                        data["scienceDislike"],
                        data["scienceCareerBenefits"],
                        data["scienceAcademicPerformance"],
                        data["scienceDifficulty"],
                        data["scienceRealWorldApplication"],
                        data["scienceCareerImportance"],
                        data["scienceActivitiesParticipation"],
                        data["techBuildingRepairAbility"],
                        data["techStudiesChoice"],
                        data["techDevicesSkills"],
                        data["techSuccessConfidence"],
                        data["techInventionsLink"],
                        data["techProjectsApplicability"],
                        data["techCuriosity"],
                        data["techCareerRelevance"],
                        data["techCoursesInterest"],
                        data["techProblemSolvingSkills"]
                        ]]
        result = model.predict(input_data)
        result_str = str(result[0])
        return jsonify({"result":result_str})
    except Exception as e:
        return jsonify({"error":str(e)})
