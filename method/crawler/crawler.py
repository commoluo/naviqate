import sys
import os

import method.utils.utils as utils
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

import undetected_chromedriver as uc

import time

from collections import defaultdict
from bs4 import BeautifulSoup
import pyautogui
import datetime
import shutil
import json

import method.utils.utils as utils
import method.llm.prompts as prompts
from method.llm.llm_prompting import *

import configparser

config = configparser.ConfigParser()
with open((r'../method/configs.config')) as config_file:
    config.read_file(config_file)
# config.readfp(open(r'../method/configs.config'))
WINDOW_WIDTH = int(config.get('DRIVER', 'width'))
WINDOW_HEIGHT = int(config.get('DRIVER', 'height'))
WAIT_TIMEOUT = int(config.get('DRIVER', 'timeout'))
MAX_ACTIONABLES = int(config.get('METHOD', 'max_actionables'))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class WebCrawler():
    def __init__(self, website, task, abstracted=False, headless=False, output_dir='./out'):
        
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f"window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}")
        chrome_options.add_experimental_option("prefs", {"profile.default_content_settings.cookies": 0})
        # chrome_options.add_argument('--headless')

        desired_caps = DesiredCapabilities.CHROME.copy()
        desired_caps["goog:loggingPrefs"] = {"browser": "ALL"}
        
        self.driver = uc.Chrome(option=chrome_options, headless=True)
        # service = Service(executable_path=ChromeDriverManager().install())
        
        self.driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # 启用网络请求监听
        self.driver.execute_cdp_cmd('Network.enable', {})
        
        self.task = task
        # self.website_category = "shopping" # We focus on e-commers websites for now
        self.website = website
        self.current_actionables = []
        self.history = []
        self.id_dict = defaultdict(lambda: 0)
        self.driver.maximize_window()
        self.driver.implicitly_wait(WAIT_TIMEOUT * 5)
        self.driver.get("https://" + website)
        self.driver.implicitly_wait(WAIT_TIMEOUT)
        self.dir = f'{output_dir}/{self.website}/{utils.string_to_filename(self.task)}'
        
        # self.close_modal()
        self.init_dir()
        # self.model = init_model(model='gpt')¡
        # self.openai_client = init_openai()
        
        # here transform the commands into vector to simpilify them
        self.semantic_similarity_model = utils.init_semantic_similarity_model()

        self.driver.switch_to.default_content()
        # here decide the version of models used
        self.model_chain = create_model_chain(init_model(model='gpt')) 
        self.model_chain_gpt_mini = create_model_chain(init_model(model='gpt_mini'))
        self.prev_context = ""

        

        if abstracted:
            self.task = self.generate_concrete_task()
            
    # def capture_request(self, request):
    #     """
    #     捕获并记录网络请求信息
    #     """
    #     log_data = {
    #         "url": request['request']['url'],
    #         "method": request['request']['method'],
    #         "headers": request['request']['headers'],
    #         "postData": request.get('request', {}).get('postData', ''),
    #         "timestamp": request['timestamp']
    #     }
    #     self.network_logs.append({"request": log_data})

    # def capture_response(self, response):
    #     """
    #     捕获并记录网络响应信息
    #     """
    #     log_data = {
    #         "status": response['response']['status'],
    #         "url": response['response']['url'],
    #         "headers": response['response']['headers'],
    #         "responseBody": self.get_response_body(response['requestId']),
    #         "timestamp": response['timestamp']
    #     }
    #     self.network_logs.append({"response": log_data})

    # def get_response_body(self, request_id):
    #     """
    #     获取响应体内容
    #     """
    #     try:
    #         response_body = self.driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
    #         return response_body['body']  # 返回响应体
    #     except Exception as e:
    #         print(f"Error fetching response body: {e}")
    #         return ""

    # def write_network_logs(self):
    #     """
    #     将网络日志写入 JSON 文件
    #     """
    #     log_file_path = os.path.join(self.dir, "network_logs.json")

    #     # 保存网络日志到文件
    #     with open(log_file_path, "w") as f:
    #         json.dump(self.network_logs, f, indent=4)

    # def clear_network_logs(self):
    #     """
    #     清空网络日志
    #     """
    #     self.network_logs = []

    # def get_network_logs(self):
    #     """
    #     获取当前捕获的所有网络日志
    #     """
    #     return self.network_logs

    # # get the console logs from chrome
    # def get_console_logs(self):
    #     try:
    #         # Each entry is a dict with keys like 'level', 'message', 'timestamp'
    #         return self.driver.get_log("browser")
    #     except:
    #         # If something goes wrong, return an empty list
    #         return []
    
    # # 获取所有的日志并保存（包括控制台日志和网络日志）
    # def write_all_logs(self):
    #     console_logs = self.get_console_logs()  # 获取控制台日志
    #     log_file_path = os.path.join(self.dir, "logs.json")

    #     # 如果日志文件存在，加载并追加新的日志
    #     if os.path.exists(log_file_path):
    #         with open(log_file_path, "r") as f:
    #             try:
    #                 existing_logs = json.load(f)
    #             except:
    #                 existing_logs = []
    #     else:
    #         existing_logs = []

    #     # 合并控制台和网络日志
    #     all_logs = {
    #         "console_logs": console_logs,
    #         "network_logs": network_logs
    #     }

    #     existing_logs.append(all_logs)

    #     # 将日志写入文件
    #     with open(log_file_path, "w") as f:
    #         json.dump(existing_logs, f, indent=4)

    def init_dir(self):

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)

        os.makedirs(self.dir, exist_ok=True)

    def generate_concrete_task(self):
        description = self.get_meta_description() 
        tasks_db = utils.init_rag()
        # generate the concrete task by retreiving and limit the maximum to 2 
        examples = utils.retreive_sample_tasks(self.semantic_similarity_model, tasks_db, self.task, n=2)
        examples = [{"abstracted_task": s["abstracted_task"], "original_task": s["original_task"]} for s in examples]
        data = {
            "Abstracted Task": self.task,
            "Website": self.website,
            "Description": description,
            "Examples": examples
        }

        data = json.dumps(data)

        response = self.model_chain(
            prompts.abstract_to_concrete_prompt,
            create_single_user_message(data)
        )
        return response

    def get_page_context(self, img, prev_state=""):
        description = self.get_meta_description()
        prev_action = ""
        # history records the action histories
        if len(self.history) > 0:
            tup = self.history[-1]
            prev_action = json.dumps({"element": tup[3], "action": tup[2]})

        data = json.dumps({
                "description": description,
                "previous_state": prev_state,
                # TODO
                # if can add the logs into a new previous_result field? 
                "previous_action": prev_action
            })
        print('page context extraction' + data)
        try:
            res = self.model_chain(
                prompts.context_extraction_prompt,
                create_multimodal_user_message(data, img)
            )
            
            print("try" + res)
        except Exception as e:
            img = self.take_screenshot()
            res = self.model_chain(
                prompts.context_extraction_prompt,
                create_multimodal_user_message(data, img)
            )
            print("exp" + res)
        finally:
            res = json.loads(utils.clean_json(res))
            
            context = f"{res['context']} {' '.join(res['sub_functionalities'])}"
            print('context in get_page_context' + context)
        return context

    # will be used in 2 different situations, for concrete task match and page context extraction
    def get_meta_description(self):
        try:
            return self.driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute("content")
        except:
            return ""

    def get_next_step(self, img):

        data = {
            "task": self.task,
            # get the element type and xPath
            "history": [(t[3], t[2]) for t in self.history], 
            "context": self.prev_context
        }

        data = json.dumps(data)

        response = self.model_chain(
                prompts.next_step_prompt,
                create_single_user_message(data)
            )

        # try:

        #     response = self.model_chain(
        #         prompts.next_step_prompt,
        #         create_multimodal_user_message(data, img)
        #     )

        # except Exception as e:
        #     img = self.take_screenshot()
        #     response = self.model_chain(
        #         prompts.next_step_prompt,
        #         create_multimodal_user_message(data, img)
        #     )

        return response

    
    def inject_actionable_ids(self):
        for i in range(len(self.current_actionables)):
            xpath = self.current_actionables[i].xpath
            elem = self.driver.find_element(By.XPATH, xpath)
            self.driver.execute_script("arguments[0].setAttribute('idx',arguments[1])", elem, i+1)

    def get_tag_name(self, object_id):
        return self.driver.execute_cdp_cmd("Runtime.callFunctionOn", {
                        "objectId": object_id,
                        "functionDeclaration": """
                            function() {
                                return this.tagName.toLowerCase();
                            }
                        """,
                        "returnByValue": True
                    })['result']['value']
    
    def extract_actionable_elements(self, tag_name, actionables):
        elements = self.driver.find_elements(By.TAG_NAME, tag_name)
        xpaths = [elem.xpath for elem in actionables]
        for element in elements:
            xpath = utils.get_xpath(self.driver, element)
            if xpath not in xpaths:
                self.driver.execute_script("arguments[0].setAttribute('idx',arguments[1])", element, len(actionables)+1)
                outerHTML = element.get_attribute('outerHTML')
                outerHTML = utils.preprocess_element(outerHTML)
                actionables.append((outerHTML, xpath, tag_name))

    def generate_id(self, tag_name):
        self.id_dict[tag_name] += 1
        return tag_name[:2] + str(self.id_dict[tag_name])

    def find_actionables(self):

        actionables = []
        # by this way can get the html tree, and the objectid is the reference of html tree
        response = self.driver.execute_cdp_cmd("Runtime.evaluate", {
            "expression": "document.querySelectorAll('body *')",
            "returnByValue": False
        })

        node_list_object_id = response['result']['objectId']

        properties = self.driver.execute_cdp_cmd("Runtime.getProperties", {
            "objectId": node_list_object_id
        })

        i = 0
        
        for property in properties['result']:
            i += 1
            try:
                if 'value' in property and 'objectId' in property['value']:
                    object_id = property['value']['objectId']
                    
                    tag_name = self.get_tag_name(object_id)

                    actionable = (tag_name in ['a', 'button', 'input', 'select']) or (property['value']['className'] in ['HTMLButtonElement', 'HTMLAnchorElement', 'HTMLInputElement', 'HTMLSelectElement'])
                    
                    # re-check if the element is actionable, method is to make sure the event listeners
                    if not actionable:
                        try:
                            listeners = self.driver.execute_cdp_cmd("DOMDebugger.getEventListeners", {"objectId": object_id})['listeners']
                            actionable = any(l['type'] == 'click' for l in listeners)
                        except Exception as e:
                            pass
                    
                    if actionable:
                        
                        try:
                            # get the xpath of the elements which constructed like '/html/body/div/p[2]' and parentnode and brothernode will be divided
                            xpath = utils.get_xpath(self.driver, object_id)
                            # print(f"Generated XPath: {xpath}")  # Debug print
                            
                            # there is problem: node info is already got in the past, but still get the element by xpath
                            # reason: 1. node info got by CDP maybe changed and unstable 2: this node info is used for selenium
                            element = self.driver.find_element(By.XPATH, xpath)

                            displayed = element.is_displayed()
                            if tag_name != 'input' and not displayed:
                                continue

                            outerHTML = self.driver.execute_cdp_cmd("DOM.getOuterHTML", {"objectId": object_id})['outerHTML']
                            # keywords = ['cookie', 'accept', 'agree']
                            # if any(keyword in outerHTML.lower() for keyword in keywords):
                            #     print(outerHTML)
                            #     element.click()
                            #     continue

                            outerHTML = utils.preprocess_element(outerHTML)
                            
                            location = self.driver.execute_script("""
                                var rect = arguments[0].getBoundingClientRect();
                                return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
                            """, element)
                            
                            actionables.append(utils.Actionable(outerHTML, xpath, tag_name, location, element))

                        # except TimeoutException:
                        #     print(f"Timeout while waiting for element with XPath: {xpath}")
                        #     continue
                        # except StaleElementReferenceException:
                        #     print(f"Stale element reference for XPath: {xpath}")
                        #     continue
                        except Exception as e:
                            # print(f"Exception while processing element with XPath: {xpath} - {e}")
                            pass

            except Exception as e:
                continue

        return actionables
    
    def annotate(self):

        for index, actionable in enumerate(self.current_actionables):

            element = actionable.element

            try:

                self.driver.execute_script("""
                    var element = arguments[0];
                    var annotation = document.createElement('div');
                    annotation.innerText = arguments[1];
                    annotation.style.position = 'absolute';
                    annotation.style.bottom = '-5px'; 
                    annotation.style.right = '-5px';
                    annotation.style.fontWeight = 'bold';
                    annotation.style.color = 'red';
                    annotation.style.backgroundColor = 'yellow';
                    annotation.style.padding = '2px';
                    annotation.style.borderRadius = '50%';
                    annotation.style.width = '30px';
                    annotation.style.height = '30px';
                    annotation.style.display = 'flex';
                    annotation.style.alignItems = 'center';
                    annotation.style.justifyContent = 'center';
                    annotation.style.fontSize = '14px';
                    annotation.style.zIndex = '1000'; 
                    annotation.id = 'custom-annotation';
                    element.style.position = 'relative';

                    element.appendChild(annotation);
                """, element, index)
            except Exception as e:
                pass

    def deannotate(self):
        self.driver.execute_script("""
            var annotations = document.querySelectorAll('#custom-annotation');
            annotations.forEach(function(annotation) {
                annotation.parentNode.removeChild(annotation);
            });
        """)


    
    def do_action(self, actionable, description, action, text):

        xpath = actionable.xpath
        
        try:
            element = self.driver.find_element(By.XPATH, xpath)
            # self.driver.implicitly_wait(0)
            # wait = WebDriverWait(self.driver, 5)
            # element = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
            # self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            # element = actionable.element
            # element.click()

            if action == 'select':
                if element.tag_name == 'select':
                    select = Select(element)
                    select.select_by_index(int(text) - 1)
                else:
                    action = 'click'

            if action == 'click':
                ActionChains(self.driver).move_to_element(element).click(element).perform()

            if action == 'type':
                element.clear()
                element.send_keys(Keys.SPACE, Keys.BACK_SPACE)
                ActionChains(self.driver).move_to_element(element).click(element).perform()
                if "checkbox" not in actionable.outerHTML:
                    element.send_keys(text)
                    element.send_keys(Keys.RETURN)

        except Exception as e:
            print(e)
            try:
                if action == 'type':
                    element.send_keys(text)
                    # element.submit()
                    element.send_keys(Keys.RETURN)
                else:
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                    element.click()
            except Exception as e:
                self.driver.implicitly_wait(WAIT_TIMEOUT)
                return 1



        if action == "type":
            self.history.append((actionable.outerHTML, xpath, f'type:{text}', description, actionable.location))
            
        if action == "select":
            self.history.append((actionable.outerHTML, xpath, f'select:{str(int(text) - 1)}', description, actionable.location))
        
        else:
            self.history.append((actionable.outerHTML, xpath, action, description, actionable.location))

        self.driver.implicitly_wait(WAIT_TIMEOUT)
        # self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + 't')
        # Switch to the new window
        self.driver.switch_to.window(self.driver.window_handles[len(self.driver.window_handles)-1])

        return 0


    def get_inner_text(self, outerHTML):
        soup = BeautifulSoup(outerHTML, 'html.parser')
        inner_text = ''
        for element in soup.find_all():
            if element.has_attr('aria-label'):
                inner_text += element['aria-label'] + ' '
            elif element.has_attr('placeholder'):
                inner_text += element['placeholder'] + ' '
            elif element.name in ['script', 'style', 'svg']:
                continue
            else:
                inner_text += element.get_text(separator=' ', strip=True) + ' '

        return inner_text.strip()
    
    def take_screenshot(self):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        filename = self.dir + "/" + timestamp + '.png'
        self.driver.get_screenshot_as_file(filename)
        img = utils.encode_image(filename)
        return img
    
    def close_modal(self):
        try:
            close_buttons = self.driver.find_elements(By.XPATH, '//button[contains(@class, "close")]')
            for button in close_buttons:
                try:
                    button.click()
                except Exception as e:
                    print("Error while clicking close button:", e)
            
            close_icons = self.driver.find_elements(By.XPATH, '//span[contains(@class, "close")]')
            for icon in close_icons:
                try:
                    icon.click()
                except Exception as e:
                    print("Error while clicking close icon:", e)


            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
            
            print("Modal closed successfully.")
        except Exception as e:
            print("Error while closing modal:", e)

    
    def accept_cookies(self):
        keywords = ['cookie', 'accept', 'agree']
        for idx, a in enumerate(self.current_actionables):
            print(a.outerHTML)
            if any(keyword in a.outerHTML.lower() for keyword in keywords):
                try:
                    print(a.outerHTML.lower())
                    self.driver.find_element(By.XPATH, a.xpath).click()
                    self.current_actionables.pop(idx)
                    return
                except:
                    pass


    def choose_action(self, data, img):
        print('choose action data:' + json.dumps(data))
        # for i, actionable in enumerate(json.dumps(data).actionables):
        #     print(f"{i}: {actionable}")
        try:
            response = self.model_chain(
                prompts.which_actionable_prompt,
                # create_single_user_message(json.dumps(data))
                create_multimodal_user_message(json.dumps(data), img)
            )
        except Exception as e:
            img = self.take_screenshot()
            response = self.model_chain(
                prompts.which_actionable_prompt,
                # create_single_user_message(json.dumps(data))
                create_multimodal_user_message(json.dumps(data), img)
            )
        try:
            response = utils.clean_json(response)
            response = json.loads(response)
            index = int(response['index'])
            action = response['action']
            text = ""
            if 'arg' in response:
                text = response['arg']

            return index, action, text
        except Exception as e:
            print(e)
            response = json.loads(response)
            index = int(response['index'])
            action = response['action']
            text = ""
            if 'arg' in response:
                text = response['arg']

            return index, action, text

    def descriptions_to_str(self, descriptions):
        descriptions_str = ""
        for i, item in enumerate(descriptions):
            descriptions_str += f"\n{i}. {item}, {self.current_actionables[i].tagName}"
        # print(descriptions_str)
        return descriptions_str

    def calculate_actionable_scores(self):
        semantic_scores = utils.get_semantic_similarity(
            self.semantic_similarity_model, 
            self.next_step, 
            [utils.get_inner_text(self.driver, a.element) or a.outerHTML for a in self.current_actionables] # get innerHTML text, which contains the semantic information
        )

        if len(self.history) == 0:
            return semantic_scores
        
        history_outerhtmls = [h[0] for h in self.history]
        history_xpaths = [h[1] for h in self.history]
        
        # distances = [a.distance_from_point(self.history[-1][4]) for a in self.current_actionables]
        
        # max_distance = max(distances)
        # min_distance = min(distances)
        # normalized_distances = [(d - min_distance) / (max_distance - min_distance) for d in distances]
        # normalized_distances = [d ** 0.1 for d in normalized_distances]
        
        # modify the score of duplicate elements
        for index, a in enumerate(self.current_actionables):
            if a.outerHTML in history_outerhtmls or a.xpath in history_xpaths:
                semantic_scores[index] *= 0.5
                
            # if normalized_distances[index] != 0:
            #     semantic_scores[index] /= normalized_distances[index]
        
        return semantic_scores



    def step(self):

        # this function is to get the actionbale elements in current HTML file 
        # at the same time do some pre-process like simplify the element(1.delete 'style' attributes  2.limit the length of html tags)
        self.current_actionables = self.find_actionables()

        if len(self.current_actionables) < 2:
            return 0

        img = self.take_screenshot()
        # get previous context, it is mostly about the description of the page
        # example: This page is the homepage for a flight tracking service. users can search for flights by flight number or route users can view live tracking maps users can check flight status and airport delays users can access information for both commercial and private flights
        self.prev_context = self.get_page_context(img, self.prev_context)

        # remove all the annotate
        self.deannotate()
    
        self.next_step = self.get_next_step(img)
        if "done" in self.next_step.lower():
            return 1

        if "back" in self.next_step.lower():
            self.driver.execute_script("window.history.go(-1)")
            self.history.pop()
            return 0

        inputs = [a for a in self.current_actionables if a.tagName == 'input']
        # this is the process of choice making, using scores to rank the 
        scores = self.calculate_actionable_scores()
        self.current_actionables, _, scores = utils.keep_top_n(self.current_actionables, self.current_actionables, scores, n=MAX_ACTIONABLES)
        for elem in inputs:
            if elem not in self.current_actionables:
                self.current_actionables.append(elem)

        # if len(self.history) > 0:

        #     distances = [a.distance_from_point(self.history[-1][4]) for a in self.current_actionables]

        #     max_distance, min_distance = max(distances), min(distances)
        #     normalized_distances = [(d - min_distance) / (max_distance - min_distance) for d in distances]

        #     self.current_actionables = [a for a, _ in sorted(zip(self.current_actionables, normalized_distances), key=lambda x: x[1])]
            
        # generate the actionable elements description by actionable element(element + xpath + context)
        descriptions = self.find_actionable_descriptions()

        self.annotate()

        descriptions_str = self.descriptions_to_str(descriptions)


        data = {
            "website": self.website,
            "actionables": descriptions_str,
            "task": self.task,
            "next_step": self.next_step,
            "history": [(t[3], t[2]) for t in self.history]
        }



        img = self.take_screenshot()
        # LMM will process both task data and image, and then give the action words
        index, action, text = self.choose_action(data, img)
        index = max(0, min(index, len(self.current_actionables) - 1))
        actionable = self.current_actionables[index]
        print("selected element: ", descriptions[index])
        feedback = self.do_action(actionable, descriptions[index], action, text)

        tries = 0

        if feedback == 1 and tries < 3:
            # retey but limit the amount, firstly remove the unactionable element and retry.
            self.current_actionables.pop(index)
            descriptions.pop(index)

            descriptions_str = self.descriptions_to_str(descriptions)

            data = {
                "website": self.website,
                "actionables": descriptions_str,
                "task": self.task,
                "next_step": self.next_step,
                "history": [(t[3], t[2]) for t in self.history]
            }

            index, action, text = self.choose_action(data, img)
            index = max(0, min(index, len(self.current_actionables) - 1))
            actionable = self.current_actionables[index]
            print("selected element: ", descriptions[index])
            feedback = self.do_action(actionable, descriptions[index], action, text)
            tries += 1

        return 0
    
    def refine_img_actionables(self):
        img_actionables = []
        for elem in self.current_actionables:
            if elem.tagName == 'img':
                self.current_actionables.remove(elem)
                img_actionables.append(elem)
        return img_actionables
    
    def get_img_descriptions(self, img_elements):
        descriptions = []
        for img in img_elements:
            elem = self.driver.find_elements(By.XPATH, img[1])[0]
            img_src = elem.get_attribute('src')
            desc = utils.run_inference_blip(self.captioner, img_src)
            if desc == None:
                self.current_actionables.append(img)
                img_elements.remove(img)
            else:
                descriptions.append(desc)

        return descriptions

    def find_actionable_descriptions(self):


        descriptions = []
    

        # img_actionables = self.refine_img_actionables()
        # img_descriptions = self.get_img_descriptions(img_actionables)


        close_elements = utils.extract_context(self.driver, self.current_actionables)
        jsons = utils.create_context_json(self.current_actionables, close_elements)

        N = 10

        chunks = [jsons[i:i+N] for i in range(0, len(self.current_actionables), N)]

        for chunk in chunks:
            prompt = utils.to_json(chunk)
            tries = 0

            while tries < 3:

                response = self.model_chain_gpt_mini(
                    prompts.element_functionality_prompt_with_context + f" Ensure that there are exactly {len(chunk)} items in the list.",
                    create_single_user_message(utils.to_json(chunk)),
                    verbose=False
                )
                try:
                    cleaned_response = utils.clean_json(response)
                    desc_list = utils.json_to_list(cleaned_response)
                    if len(chunk) != len(desc_list):
                        tries += 1
                        continue
                    break

                except Exception as e:
                    tries += 1
                    continue

            if tries == 3:
                print('LLM Error - Abort')
                self.quit()

            descriptions.extend(desc_list)
            


        for index, a in enumerate(self.current_actionables):
            if a.tagName == 'select':
                select_element = Select(self.driver.find_element(By.XPATH, a.xpath))
                options = [f"{i+1}_{utils.get_inner_text(self.driver, s)}" for i, s in enumerate(select_element.options)]
                descriptions[index] += " with these options: " + ', '.join(options)
                
        return descriptions

    
    def history_to_json(self):

        data_list = []

        for tup in self.history:
            elem, xpath, action, _, _ = tup
            data_list.append({"element": elem, "xpath": xpath, "action": action})

        json_data = json.dumps(data_list, indent=4)

        filepath = self.dir + "/"

        with open(filepath + "history.json", "w") as json_file:
            json_file.write(json_data)

    def quit(self):
        img = self.take_screenshot()
        self.history_to_json()
        self.driver.close()
        self.driver.quit()

    def loop(self, MAX_STEPS=15):

        for i in range(MAX_STEPS):
            print('step index:' + str(i))
            try:
                agree_elements = self.driver.find_elements(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'close')]")

                for element in agree_elements:
                    element.click()
                # 等待并定位 Accept All Cookies 按钮
                accept_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept All Cookies')]"))
                )
                # 使用 JavaScript 确保点击成功
                self.driver.execute_script("arguments[0].click();", accept_button)
                print("Cookies bar closed successfully.")
            except Exception as e:
                # print(e)
                pass
            if self.step() == 1:
                break
        self.quit()