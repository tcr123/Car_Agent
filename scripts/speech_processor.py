#!/usr/bin/env python
from shutil import ExecError
from urllib import response
import rospy
from std_msgs.msg import String
from openai import OpenAI
import time

GROQ_API_KEY="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"
GROQ_BASE_URL="https://api.groq.com/openai/v1"
GROQ_MODEL="mixtral-8x7b-32768"

SYSTEM_PROMPT = """You are an automobile seller. \
Given a user query and some information about a car model, \
you need to provide a consise and professional response to the user query. \
If you don't know the answer, just say that you don't know. \
Keep the answer short and concise. Answer in a natural and attractive way. You do not need to use every car information provided. \

Car information: \
{car_info}
"""

class SpeechProcessor:
    def __init__(self):
        self.speech_received = False
        self.car_received = False
        self.speech_data = None
        self.car_data = None
        self.is_running = False

        rospy.loginfo("Initializing SpeechProcessor")
        rospy.init_node('speech_processor', anonymous=True)

        rospy.Subscriber("result", String, self.process_speech)
        rospy.Subscriber("car_recognition_output", String, self.update_car_data)
        self.response_pub = rospy.Publisher('car_recognition_response', String, queue_size=10)

        self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    def update_car_data(self, data):
        if str(data.data).strip() and not self.is_running:
            self.car_data = data.data
            self.car_received = True
            rospy.loginfo("Updated car data: %s", self.car_data)

    def process_speech(self, data):
        if not data.data or not self.car_data or self.is_running:
            return
        self.speech_data = data.data
        self.speech_received = True
        rospy.loginfo("Received speech: %s", self.speech_data)
        self.try_process()
        
    def try_process(self):
        if self.speech_received and self.car_received:
            rospy.loginfo("Processing speech and car data")
            response = self.answer(self.speech_data, self.car_data)
            rospy.loginfo(f"Generated response: {response}")
            self.publish_response(response)

            self.speech_received = False
            self.car_received = False
            self.car_data = None
            self.speech_data = None
            self.is_running = True
            time.sleep(2)
            self.is_running = False

    def publish_response(self, response):
        self.response_pub.publish(response)
        rospy.loginfo(f"Published response!")

    def answer(self, user_prompt, car_info, temperature = 0.3, top_p = 1.0):

        system_prompt = SYSTEM_PROMPT.replace("{car_info}", str(car_info))

        # rospy.loginfo(f"Car info: {system_prompt}")
        # rospy.loginfo(f"User query: {user_prompt}")

        response_text = ""
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=temperature,
            top_p=top_p
        )

        response_text = response.choices[0].message.content
        
        return response_text

def main():
    processor = SpeechProcessor()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt")
        pass
    except Exception as e:
        rospy.loginfo(f"An error occured: {e}")