import logging, time
import sys, json
import pyizumo
import multiprocessing as mp
from multiprocessing import Process, Manager, Queue
import queue

# nlp = pyizumo.load("en")
#from libSIRE.Tokenizer import Tokenizer

manager = Manager()

d = manager.dict()
tasks_to_accomplish = Queue()
result = Queue()

def tokenize(inqueue, d):
    pid = mp.current_process()
    tok = pyizumo.load("en", parsers=['token', 'sentence'])
    while True:
        try:
            print(f"pid {pid} size: {inqueue.qsize()}")
            id, text = inqueue.get(block=True, timeout=0.05)
        except queue.Empty:
            # print(f"{pid}: empty queue")
            break
        except Exception as e:
            # print(f"Got exception: {e}")
            break
        else:
            # print(f"{pid} Processing {id}")
            tokens = []
            proc_text = tok(text)#  nlp(text)
            # proc_text = tok(text)  # nlp(text)
            for sent in proc_text.sentences:
                tokens.append([t.text for t in sent.tokens])
            # outqueue.put(json.dumps({"id": id, "tokens": tokens}))
            d[id] = tokens
            time.sleep(0.01)
    print(f"{pid} exiting.")
    return True


texts = ["Time Event Confirmation for Secondary Resources  Use  You use this function to confirm time events for the secondary resources of a process order by sending a corresponding process message in process control. You can confirm the following time events:   Processing time events for particular points in time (for example, start, finish)  Time events for variable activities  See also:   Completion Confirmations in the Process Order   Prerequisites  Standard values with record type group Processing or Variable activity have been maintained in the standard value key in the resource master.   The control key that has been assigned to the secondary resource allows confirmation.  The process order has been released.  Features  You can use the following process messages and destinations to confirm time events:   Time event   Message category   Destination   Processing time event   PI_SRST   PI05   Time event for variable activity   PI_SRACT   PI11   The following sections describe what you must bear in mind when you create process messages.   Process Messages for Processing Time Events   Time event:   You determine which time event you want to confirm by entering the corresponding resource status in the process message. You can confirm the following time events or statuses:  Time event / status   Meaning   Start   Partial confirmation: Resource usage has started. You must also set this status when you continue using the secondary resource after an interruption.   Finish   Final confirmation: Resource usage has finished.   Interruption   Partial confirmation: Resource usage has been interrupted.   Partial finish   Partial confirmation: Resource usage has not yet finished, but the activity performed so far is to be determined.   In the process message, you specify the confirmed point in time using the characteristics for the event date and time.   Resource:   You determine the secondary resource for which you confirm time events by including the characteristics used for the process order, operation or phase, and item number of the secondary resource in the process message.  For the time event Start, you can also confirm that a different resource was used than planned. To do so, you include the characteristics used for the resource name and plant of the resource in the process message.  Process Message for Variable Activities   Time event:   To determine the time event you want to confirm, you enter the corresponding status in the process message. You can confirm the following time events or statuses:  Time event / status   Meaning   Partial finish   Partial confirmation: Resource usage has not yet finished, but the activity performed so far is to be determined.   Finish   Final confirmation: Resource usage has finished.   If no specific time event is contained in the process message, the system automatically sets status Partial finish.   Resource:   As for processing time events, you determine the secondary resource for which you confirm time events by including the characteristics used for the process order, operation or phase, and item number of the secondary resource in the process message.  For time events for variable activities however, you cannot confirm resources deviating from the planned data.  Variable activity:   You use the parameter ID of the standard value to determine the activity you want to confirm in the process message. The parameter must be defined in the standard value key of the secondary resource.  You enter the activity performed and, if required, the unit of measurement for this parameter in the process message.  General Notes on Message Processing   The following applies to both processing time events and variable activities.   Additional confirmation data:   If required, you can also confirm a short text for the confirmation and the reason for deviating from the planned data.  Consistency checks:   When processing the message, the system checks whether the values contained in it are valid and consistent with one another. If an inconsistency is discovered (for example, an invalid order number or a time event that is not allowed at the current processing stage of the phase):  The system writes a corresponding error or warning message to the message log  The system sets the status of the process message to Destination erro r or Sent with warning   Subsequent processing:   All further processing of the data confirmed is carried out according to the general logic of the confirmation function.  Confirmation number:   The system assigns a unique number to each confirmation. The number is written to the message log.",
         "Report: Classifications for Force Element   Use  You can use this report to determine and display the classifications of a force element defined using the classification system of the SAP system.  Prerequisites  You have classified force elements. See also Classification .   Features  The following selection parameters are available:   Force Element   You specify the required force element for which you want to determine classification data.  From/To   You specify the required evaluation period.  Individual Operation / Structure Operation   You specify whether you want to evaluate only the specified force element or its subordinate force elements as well.  Use   You specify which structures you want to evaluate.  Class Selection   You specify which classes you want to evaluate.  Characteristics and Evaluation   You specify whether you want to evaluate Text Characteristics , Numeric Characteristics , and/or Currency Characteristics .   Activities  Choose report /ISDFPS/DFOBJECT_CLASS_ASSIGN.  Enter the required selection criteria.  Choose Execute. ",
         "SLCM Business Function for the Admissions Portal \u00c2\u00a0 Technical Data Technical Name of Business Function ISHERCM_ADMISSION_PORTAL Type of Business Function Industry Business Function Available From SAP S/4HANA, on-premise edition 1511 FPS01 Application Component Student Lifecycle Management (IS-HER-CM) for S/4HANA: IS-PS-CA 800 SP 11 Required Business Function ISHERCM_MAIN: Academic Structure ISHERCM_SETTINGS - Customizing and Settings ISHERCM_ADM_DECISION - Admission Decision Incompatible Business Function Not relevant Reversible Yes   You can use this business function to create admission applications for use by prospective applicants to your higher education institution. These applications in turn can be further processed to meet your selection requirements using a related decision framework application. Applicants can then use this runtime online form created to complete and submit applications directly using your institution's website. Prerequisites You have installed the following components as of the version mentioned: Type of Component Component Required for the Following Features Only Software Component Student Lifecycle ManagementIS-PS-CA 800   You have activated the following business functions: ISHERCM_MAIN: Academic Structure  ISHERCM_SETTINGS - Customizing and Settings  Features Admissions Application Master Data Create and maintain master data for your admissions department, such as, organizational units, programs of study, qualifications, and course offerings. Admission Procedures Create an admissions procedure for each course offering in your course catalog. This can then be further processed using the connected admissions decision framework application. Create Applicant Questionnaires using the Application Form Designer Once you have created suitable admissions procedures for your course offerings, you can use the use the Generic Application Form Designer to create two kinds of questionnaires for a prospective applicants, a preliminary questionnaire to gather basic information about the applicant, and a more specific actual questionnaire, which is designed based on the applicant's chosen course offering. Create Application Forms using Application Form Designer You can create multiple customized admissions application forms to meet the admissions needs of your institution using the transaction PIQ_FDCONFIG. Online Portal Student Applications Applicants can directly access the admissions portal through your higher education institution web site. Existing user can log on directly and start a new application or edit and submit draft applications, together with necessary supporting documents. Submitting an applicant triggers a process in the backend which creates a new student user. The application is then processed in accordance with the admission decision framework configured by the administrator for the chosen course offering. More Information For more information on Admissions Procedure Decision Management in Student Lifecycle Management see: Business function SLCM Business Function for Decision Management"]*1000

for i, t in enumerate(texts):
    tasks_to_accomplish.put([i, t])

from docuverse.utils.timer import timer
tm = timer("Processing time.")
    
res = {}
num_processes = 20
# num_processes = 1
processes = []
tm.mark()
for i in range(num_processes):
    p = Process(target=tokenize, args=(tasks_to_accomplish, d))
    processes.append(p)
    p.start()

for i, p in enumerate(processes):
    p.join()
    print(f"Process {i} is done.")
print(f"Processing time: {tm.mark_and_return_time()}")
print("done processing.")
# for id, val in d.items():
#     print(f"{id}: {val}")
# while not result.empty():
#     print(result.get())
print("Done.")
