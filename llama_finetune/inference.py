import argparse
import torch
import pickle
from transformers import LlamaTokenizer, GenerationConfig
from model_utils import load_model
from prompt import format_prompt
from datasets import load_dataset
import json

def parse_args():
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument("--dataset_path", type=str, help="the dataset path where the json is stored")
    parser.add_argument("--base_model_name", type=str, help="the base model name")
    parser.add_argument('--output_path', type=str, help='path to store results')

    # inference arguments
    parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling softmax temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="The maximum number of tokens to generate")
    parser.add_argument("--include_example", type=bool, default=False, help="Include reasoning example")
    args = parser.parse_args()
    return args


class LlamaInference:
  def __init__(self, args):
    self.model_name = args['base_model_name']
    self.dataset_path = args['dataset_path']

    self.generation_config = GenerationConfig(
      do_sample = args['do_sample'],
      temperature = args['temperature'],
      top_p = args['top_p'],
      num_return_sequences = args['num_return_sequences'])
  

    self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model = load_model(self.model_name, True)
    self.model.eval()

  def generate(self, input_string, **generate_kwargs):
    # call tokenizer
    inputs = self.tokenizer(input_string, return_tensors="pt").to(device='cuda')

    with torch.no_grad():
        generate_ids = self.model.generate(**inputs, **generate_kwargs,
                                         generation_config=self.generation_config)
        # only output the generated tokens
        input_length = 1 if self.model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        generate_ids = generate_ids[:, input_length:]
    return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def run_inference(runner, dataset, outfile_path, examples=None, **generator_kwargs):
  with open(outfile_path, 'w') as f:
    for i, d in enumerate(dataset):
        prompt = format_prompt(d)
        if examples is not None:
            prompt = f"{fin_examples}\n{prompt}"
        out = runner.generate(prompt, **generator_kwargs)
        f.write(json.dumps({
            'id': d['id'],
            'output': out
        }))
        f.write('\n')  

fin_examples = '''
### INSTRUCTION:
Read the following table and answer a question

Table:
||year|gallons|average priceper gallon|aircraft fuel expense|percent of total operating expenses||
||2018|4447|$ 2.23|$ 9896|23.6% ( 23.6 % )||
||2017|4352|1.73|7510|19.6% ( 19.6 % )||
||2016|4347|1.42|6180|17.6% ( 17.6 % )||

Context:
the following table shows annual aircraft fuel consumption and costs , including taxes , for our mainline and regional operations for 2018 , 2017 and 2016 ( gallons and aircraft fuel expense in millions ) ..year gallons average price per gallon aircraft fuel expense percent of total operating expenses .
as of december 31 , 2018 , we did not have any fuel hedging contracts outstanding to hedge our fuel consumption ..as such , and assuming we do not enter into any future transactions to hedge our fuel consumption , we will continue to be fully exposed to fluctuations in fuel prices ..our current policy is not to enter into transactions to hedge our fuel consumption , although we review that policy from time to time based on market conditions and other factors ..fuel prices have fluctuated substantially over the past several years ..we cannot predict the future availability , price volatility or cost of aircraft fuel ..natural disasters ( including hurricanes or similar events in the u.s ..southeast and on the gulf coast where a significant portion of domestic refining capacity is located ) , political disruptions or wars involving oil-producing countries , economic sanctions imposed against oil-producing countries or specific industry participants , changes in fuel-related governmental policy , the strength of the u.s ..dollar against foreign currencies , changes in the cost to transport or store petroleum products , changes in access to petroleum product pipelines and terminals , speculation in the energy futures markets , changes in aircraft fuel production capacity , environmental concerns and other unpredictable events may result in fuel supply shortages , distribution challenges , additional fuel price volatility and cost increases in the future ..see part i , item 1a ..risk factors 2013 201cour business is very dependent on the price and availability of aircraft fuel ..continued periods of high volatility in fuel costs , increased fuel prices or significant disruptions in the supply of aircraft fuel could have a significant negative impact on our operating results and liquidity . 201d seasonality and other factors due to the greater demand for air travel during the summer months , revenues in the airline industry in the second and third quarters of the year tend to be greater than revenues in the first and fourth quarters of the year ..general economic conditions , fears of terrorism or war , fare initiatives , fluctuations in fuel prices , labor actions , weather , natural disasters , outbreaks of disease and other factors could impact this seasonal pattern ..therefore , our quarterly results of operations are not necessarily indicative of operating results for the entire year , and historical operating results in a quarterly or annual period are not necessarily indicative of future operating results ..domestic and global regulatory landscape general airlines are subject to extensive domestic and international regulatory requirements ..domestically , the dot and the federal aviation administration ( faa ) exercise significant regulatory authority over air carriers ..the dot , among other things , oversees domestic and international codeshare agreements , international route authorities , competition and consumer protection matters such as advertising , denied boarding compensation and baggage liability ..the antitrust division of the department of justice ( doj ) , along with the dot in certain instances , have jurisdiction over airline antitrust matters. .

Question:
what was the total operating expenses in 2018 in millions

### Response:
Explanation: The aircraft fuel expense in 2018 was 9896, which was 23.6% of the total operating expenses. Hence the total operating expense is 9896/23.6*100=41932.

The answer is 41932
### End

### INSTRUCTION:
Read the following table and answer a question

Table:
||( in millions )|dec 28 2013|dec 29 2012||
||available-for-sale investments|$ 18086|$ 14001||
||cash|854|593||
||equity method investments|1038|992||
||loans receivable|1072|979||
||non-marketable cost method investments|1270|1202||
||reverse repurchase agreements|800|2850||
||trading assets|8441|5685||
||total cash and investments|$ 31561|$ 26302||

Context:
the fair value of our grants receivable is determined using a discounted cash flow model , which discounts future cash flows using an appropriate yield curve ..as of december 28 , 2013 , and december 29 , 2012 , the carrying amount of our grants receivable was classified within other current assets and other long-term assets , as applicable ..our long-term debt recognized at amortized cost is comprised of our senior notes and our convertible debentures ..the fair value of our senior notes is determined using active market prices , and it is therefore classified as level 1 ..the fair value of our convertible long-term debt is determined using discounted cash flow models with observable market inputs , and it takes into consideration variables such as interest rate changes , comparable securities , subordination discount , and credit-rating changes , and it is therefore classified as level 2 ..the nvidia corporation ( nvidia ) cross-license agreement liability in the preceding table was incurred as a result of entering into a long-term patent cross-license agreement with nvidia in january 2011 ..we agreed to make payments to nvidia over six years ..as of december 28 , 2013 , and december 29 , 2012 , the carrying amount of the liability arising from the agreement was classified within other accrued liabilities and other long-term liabilities , as applicable ..the fair value is determined using a discounted cash flow model , which discounts future cash flows using our incremental borrowing rates ..note 5 : cash and investments cash and investments at the end of each period were as follows : ( in millions ) dec 28 , dec 29 .
in the third quarter of 2013 , we sold our shares in clearwire corporation , which had been accounted for as available-for-sale marketable equity securities , and our interest in clearwire communications , llc ( clearwire llc ) , which had been accounted for as an equity method investment ..in total , we received proceeds of $ 470 million on these transactions and recognized a gain of $ 439 million , which is included in gains ( losses ) on equity investments , net on the consolidated statements of income ..proceeds received and gains recognized for each investment are included in the "available-for-sale investments" and "equity method investments" sections that follow ..table of contents intel corporation notes to consolidated financial statements ( continued ) .

Question:
what percentage of total cash and investments as of dec 29 2012 was comprised of available-for-sale investments?

### Response:
Explanation: The total cash and investment as of dec 29 2012 was 26302. The available-for-sale investments as of dec 29 2012 was 14001. Hence the percentage is 14001/26302*100%=53%

The answer is 53%
### End

### INSTRUCTION:
Read the following table and answer a question

Table:
|||amount ( in millions )||
||2007 net revenue|$ 991.1||
||retail electric price|-17.1 ( 17.1 )||
||purchased power capacity|-12.0 ( 12.0 )||
||net wholesale revenue|-7.4 ( 7.4 )||
||other|4.6||
||2008 net revenue|$ 959.2||

Context:
entergy louisiana , llc management's financial discussion and analysis net revenue 2008 compared to 2007 net revenue consists of operating revenues net of : 1 ) fuel , fuel-related expenses , and gas purchased for resale , 2 ) purchased power expenses , and 3 ) other regulatory charges ..following is an analysis of the change in net revenue comparing 2008 to 2007 ..amount ( in millions ) .
the retail electric price variance is primarily due to the cessation of the interim storm recovery through the formula rate plan upon the act 55 financing of storm costs and a credit passed on to customers as a result of the act 55 storm cost financing , partially offset by increases in the formula rate plan effective october 2007 ..refer to "hurricane rita and hurricane katrina" and "state and local rate regulation" below for a discussion of the interim recovery of storm costs , the act 55 storm cost financing , and the formula rate plan filing ..the purchased power capacity variance is due to the amortization of deferred capacity costs effective september 2007 as a result of the formula rate plan filing in may 2007 ..purchased power capacity costs are offset in base revenues due to a base rate increase implemented to recover incremental deferred and ongoing purchased power capacity charges ..see "state and local rate regulation" below for a discussion of the formula rate plan filing ..the net wholesale revenue variance is primarily due to provisions recorded for potential rate refunds related to the treatment of interruptible load in pricing entergy system affiliate sales ..gross operating revenue and , fuel and purchased power expenses gross operating revenues increased primarily due to an increase of $ 364.7 million in fuel cost recovery revenues due to higher fuel rates offset by decreased usage ..the increase was partially offset by a decrease of $ 56.8 million in gross wholesale revenue due to a decrease in system agreement rough production cost equalization credits ..fuel and purchased power expenses increased primarily due to increases in the average market prices of natural gas and purchased power , partially offset by a decrease in the recovery from customers of deferred fuel costs. .

Question:
what is the growth rate in net revenue in 2008?

### Response:
Explanation: The net revenue in 2008 was 959.2. The net revenue the year before in 2007 was 991.1. Hence the net growth in 2018 was 959.2-991.1=-31.9. The net growth is -31.9/991.1*100%=-3.2%

The answer is -3.2%
### End'''

if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset('json', data_files=args.dataset_path, split='train')
    inf = LlamaInference({
        'dataset_path': '',
        'base_model_name': '/hpctmp/e0293904/Llama-2-7b-hf',
        'do_sample': True,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'num_return_sequences': args.num_return_sequences,
    })
    examples = fin_examples if args.include_example else None
    run_inference(inf, dataset, args.output_path, examples=examples, repetition_penalty=1.05, penalty_alpha=0.6, top_k=4, max_new_tokens=args.max_new_tokens)

