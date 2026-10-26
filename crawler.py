# -*- coding: utf-8 -*-
# 此代码仅用于学习交流使用，严禁用于非法用途                
import requests
import json, ssl, time, math, warnings, os, re


ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")



def download_file(file_path, download_url, headers={}, check_exists=True, cookies={}):
    # 下载文件
    print('*' * 100)
    if os.path.exists(file_path) and check_exists:
        print(file_path, '文件已存在')
        return True
    print(f"保存路径：{file_path}")
    rindex1 = file_path.rfind(os.path.sep)
    rindex2 = file_path.rfind('/')
    rindex = rindex1 if rindex1 > rindex2 else rindex2
    file_folder = file_path[0:rindex + 1]
    # print(file_folder)
    if not os.path.exists(file_folder) and file_folder != '':
        os.makedirs(file_folder)
    # print(f'下载URL：{download_url}')
    try:
        response = requests.get(url=download_url, headers=headers, stream=True, timeout=600, verify=False,
                                cookies=cookies)
        # response.encoding = 'utf-8'
    except Exception as e:
        print(str(e))
        return False
    else:
        try:
            if not response.status_code == 200:
                print(download_url, f'响应码为{response.status_code}')
                return None
            # print(response.headers)
            if not "content-length" in response.headers.keys():
                # print(download_url, '下载失败')
                # return None
                content_size = len(response.content)
            else:
                content_size = int(response.headers["content-length"])
            size = 0
            with open(file_path, "ab+") as file:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    file.flush()
                    size += len(data)
                    print("\r文件下载进度:%d%%(%0.2fMB/%0.2fMB)" % (
                        float(size / content_size * 100), (size / 1024 / 1024),
                        (content_size / 1024 / 1024)),
                          end=" ")
            print()
            return True
        except:
            return False



class DzSpider(object):
    def __init__(self):
        # 过程数据缓存
        self.data = {}
        # 数据存储目录
        self.folder = fr'{os.getcwd()}'
        # 数据库
        self.db_file = fr'{self.folder}\data.db'
        # 开始页码
        self.start_page = 1
        # 结束页码
        self.end_page = 10
        # 采集数量标识
        self.spider_num = 1
        # 页码大小
        self.page_size = 20
        # 是否已结束
        self.has_finish = False
        # 是否自动获取页码
        self.reset_end_page = True
        # 请求头信息
        self.headers_str = '''
        Accept: application/json, text/javascript, */*; q=0.01
		Accept-Language: zh-CN,zh;q=0.9,en;q=0.8
		Cache-Control: no-cache
		Connection: keep-alive
		Content-Type: application/x-www-form-urlencoded; charset=UTF-8
		Origin: https://digicol.dpm.org.cn
		Pragma: no-cache
		Referer: https://digicol.dpm.org.cn/?page=2&category=17
		Sec-Fetch-Dest: empty
		Sec-Fetch-Mode: cors
		Sec-Fetch-Site: same-origin
		sec-ch-ua: "Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"
		sec-ch-ua-mobile: ?0
		sec-ch-ua-platform: "Windows"
		User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36
		X-Requested-With: XMLHttpRequest
		Cookie: sajssdk_2015_cross_new_user=1; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22198e8b51a382ac-0c81dc7645cfd2-26011051-2073600-198e8b51a39114a%22%2C%22first_id%22%3A%22%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTk4ZThiNTFhMzgyYWMtMGM4MWRjNzY0NWNmZDItMjYwMTEwNTEtMjA3MzYwMC0xOThlOGI1MWEzOTExNGEifQ%3D%3D%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%22%2C%22value%22%3A%22%22%7D%7D; _abfpc=88bdffe93cd5efd1488d4bfb2c445246c75cb9cd_2.0; cna=153a4fb8d2a2d0ed8c45e6d29d744b7d; ff597e398d0138469a=OTQ2MWUyNjItMWU0YS00ZGEwLWI2YmYtNDFjZDVmZGZmMGY4
        '''
        self.headers = dict(
            [[y.strip() for y in x.strip().split(':', 1)] for x in self.headers_str.strip().split('\n') if x.strip()])

    def run_task(self):
        # 遍历所有的页码，采集数据
        page_index = self.start_page
        self.end_page = 1000
        self.reset_end_page = True
        self.has_finish = False
        while page_index < self.end_page + 1 and not self.has_finish:
            self.get_one_page(page_index)
            page_index += 1
            time.sleep(3)

    def get_one_page(self, page_index):
        req_url = 'https://digicol.dpm.org.cn/cultural/queryList'
        data = {
            "page": page_index,
            "keyWord": "",
            "cateList": "17",
            "sortType": "",
            "hasImage": "False",
            "topicId": "",
            "ancientText": "0",
            "ranNum": "0.13587795822122717"
        }

        params = {}
        res = requests.post(req_url, headers=self.headers, data=data, params=params, verify=False).json()
        if self.reset_end_page:
            self.end_page = math.ceil(res.get('recordcount') / self.page_size)
            self.reset_end_page = False

        for item in res.get('rows'):
            # if self.spider_num > 2000:
            #     self.has_finish = True
            #     break
            centerImage = item.get('centerImage')
            name = item.get('name')
            print(f'========================================\n'
                  f'第{page_index}/{self.end_page}页，总第{self.spider_num}条\n'
                  f'{name}\n'
                  f'{centerImage}\n'
                  f'========================================\n')
            save_name = fr'{self.folder}\images\{self.spider_num}_{name}.png'
            download_file(save_name, centerImage)
            self.spider_num += 1


if __name__ == '__main__':
    spider = DzSpider()
    spider.run_task()
