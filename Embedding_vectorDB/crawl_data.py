import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from urllib.parse import urljoin
import re


class DXGovCrawlerWithEmbedding:
    def __init__(self, output_dir="van_ban_downloads"):
        self.base_url = "https://dx.gov.vn"
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_total_pages(self):
        """Lấy tổng số trang bằng cách tìm trang cuối cùng"""
        url = f"{self.base_url}/van-ban-trang-1.htm"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            pagination = soup.find('ul', class_='pagination')
            if not pagination:
                print("Không tìm thấy pagination, thử tìm theo cách khác...")
                return self._find_last_page_by_testing()

            max_page = 1
            links = pagination.find_all('a')

            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)

                match = re.search(r'trang-(\d+)\.htm', href)
                if match:
                    page_num = int(match.group(1))
                    max_page = max(max_page, page_num)

                if text.isdigit():
                    page_num = int(text)
                    max_page = max(max_page, page_num)

            next_btn = pagination.find('a', string='»')
            if next_btn and max_page > 0:
                print(f"Tìm thấy {max_page} trang trong pagination, đang kiểm tra thêm...")
                actual_max = self._find_last_page_by_testing(start_page=max_page)
                max_page = max(max_page, actual_max)

            return max_page

        except Exception as e:
            print(f"Lỗi khi lấy số trang: {e}")
            return self._find_last_page_by_testing()

    def _find_last_page_by_testing(self, start_page=1):
        """Tìm trang cuối bằng cách test từng trang"""
        print("Đang tìm trang cuối bằng binary search...")

        current = start_page
        step = 10

        while current <= 500:
            url = f"{self.base_url}/van-ban-trang-{current}.htm"
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    table = soup.find('table')
                    if table and len(table.find_all('tr')) > 1:
                        print(f"  Trang {current} tồn tại ✓")
                        current += step
                    else:
                        print(f"  Trang {current} không có dữ liệu ✗")
                        break
                else:
                    print(f"  Trang {current} không tồn tại ✗")
                    break
            except:
                print(f"  Lỗi khi kiểm tra trang {current}")
                break

        if current > start_page + step:
            low = current - step
            high = current - 1

            print(f"Binary search từ trang {low} đến {high}...")

            while low <= high:
                mid = (low + high) // 2
                url = f"{self.base_url}/van-ban-trang-{mid}.htm"

                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        table = soup.find('table')
                        if table and len(table.find_all('tr')) > 1:
                            low = mid + 1
                        else:
                            high = mid - 1
                    else:
                        high = mid - 1
                except:
                    high = mid - 1

            return high

        return max(1, current - step)

    def crawl_page(self, page_num=1):
        """Crawl một trang văn bản"""
        url = f"{self.base_url}/van-ban-trang-{page_num}.htm?Field=0&Agency=0&Type=0&keyword="

        print(f"Đang crawl trang {page_num}: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            table = soup.find('table')
            if not table:
                print(f"Không tìm thấy bảng dữ liệu ở trang {page_num}")
                return []

            documents = []
            rows = table.find_all('tr')[1:]

            if not rows:
                print(f"Trang {page_num} không có dữ liệu")
                return []

            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 6:
                    doc = {
                        'so_ky_hieu': cols[0].get_text(strip=True),
                        'loai_van_ban': cols[1].get_text(strip=True),
                        'linh_vuc': cols[2].get_text(strip=True),
                        'trich_yeu': cols[3].get_text(strip=True),
                        'ngay_ban_hanh': cols[4].get_text(strip=True),
                        'download_link': None
                    }

                    download_td = cols[5]
                    download_link = download_td.find('a')
                    if download_link and download_link.get('href'):
                        doc['download_link'] = urljoin(self.base_url, download_link['href'])

                    documents.append(doc)

            print(f"✓ Trang {page_num}: Tìm thấy {len(documents)} văn bản")
            return documents

        except Exception as e:
            print(f"✗ Lỗi khi crawl trang {page_num}: {e}")
            return []

    def get_file_extension(self, url, content_type=None):
        """Xác định đúng extension của file"""
        url_ext = os.path.splitext(url.lower())[1]
        if url_ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar']:
            return url_ext

        if content_type:
            content_type = content_type.lower()
            if 'pdf' in content_type:
                return '.pdf'
            elif 'msword' in content_type or 'document' in content_type:
                return '.doc'
            elif 'wordprocessingml' in content_type:
                return '.docx'
            elif 'ms-excel' in content_type or 'spreadsheet' in content_type:
                if 'sheet' in content_type:
                    return '.xlsx'
                return '.xls'
            elif 'zip' in content_type:
                return '.zip'

        return '.pdf'

    def download_file(self, url, base_filename):
        """Tải xuống file văn bản với extension đúng"""
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('Content-Type', '')

            ext = self.get_file_extension(url, content_type)

            safe_name = re.sub(r'[^\w\-.]', '_', base_filename)
            safe_name = os.path.splitext(safe_name)[0]
            filename = f"{safe_name}{ext}"

            print(f"Đang tải: {filename}")

            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()

            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = os.path.getsize(filepath)
            print(f"✓ Đã tải: {filename} ({file_size / 1024:.1f} KB)")
            return True, filename, filepath

        except Exception as e:
            print(f"✗ Lỗi khi tải {base_filename}: {e}")
            return False, None, None

    def process_document_api(self, file_path):
        """Gọi API process-document để chuyển file thành markdown"""
        try:
            print(f"   📄 Đang xử lý document: {os.path.basename(file_path)}")

            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                response = requests.post(
                    f"http://localhost:8000/api/v1/process-document",
                    files=files,
                    timeout=60
                )

            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Process document thành công")
                return result.get('markdown_content'), None
            else:
                print(f"   ✗ Process document thất bại: {response.text}")
                return None, f"API error: {response.status_code}"

        except Exception as e:
            print(f"   ✗ Lỗi khi gọi process-document API: {e}")
            return None, str(e)

    def embed_markdown_api(self, markdown_content, document_id):
        """Gọi API embed-markdown để tạo embeddings và lưu vào vector DB"""
        try:
            print(f"   🔗 Đang tạo embeddings cho document: {document_id}")

            payload = {
                "markdown_content": markdown_content,
                "document_id": document_id,
                "chunk_mode": "sentence"
            }

            response = requests.post(
                f"http://localhost:8000/api/v1/embed-markdown",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Embedding thành công: {result.get('stored_count')} chunks")
                return True, result
            else:
                print(f"   ✗ Embedding thất bại: {response.text}")
                return False, f"API error: {response.status_code}"

        except Exception as e:
            print(f"   ✗ Lỗi khi gọi embed-markdown API: {e}")
            return False, str(e)

    def delete_document_embeddings(self, document_id):
        """
        Xóa tất cả embeddings của một document_id

        Args:
            document_id: ID của document cần xóa

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            print(f"   🗑️  Đang xóa embeddings cho document: {document_id}")

            response = requests.delete(
                f"http://localhost:8000/api/v1/document/delete/{document_id}",
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Xóa thành công: {result.get('message')}")
                return True, result.get('message', 'Document deleted successfully')
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                print(f"   ✗ Xóa thất bại: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = f"Lỗi khi gọi delete API: {str(e)}"
            print(f"   ✗ {error_msg}")
            return False, error_msg

    def delete_embeddings_from_folder(self, folder_path=None):
        """
        Xóa tất cả embeddings dựa trên các file PDF/DOC/DOCX trong thư mục

        Args:
            folder_path: Đường dẫn thư mục chứa file (mặc định: output_dir)

        Returns:
            dict: Thống kê kết quả xóa
        """
        if folder_path is None:
            folder_path = self.output_dir

        if not os.path.exists(folder_path):
            print(f"❌ Không tìm thấy thư mục: {folder_path}")
            return {
                "success": False,
                "error": "Folder not found",
                "total": 0,
                "deleted": 0,
                "failed": 0
            }

        try:
            print("=" * 60)
            print("BẮT ĐẦU XÓA EMBEDDINGS TỪ THỦ MỤC FILE")
            print("=" * 60)

            # Tìm tất cả file PDF, DOC, DOCX, XLS, XLSX
            supported_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
            all_files = []

            for ext in supported_extensions:
                files = [f for f in os.listdir(folder_path)
                         if f.lower().endswith(ext)]
                all_files.extend(files)

            total_files = len(all_files)

            if total_files == 0:
                print(f"⚠️  Không tìm thấy file nào trong thư mục: {folder_path}")
                return {
                    "success": True,
                    "total": 0,
                    "deleted": 0,
                    "failed": 0,
                    "message": "No files to delete"
                }

            print(f"\n📊 Tìm thấy {total_files} files trong thư mục")
            print(f"📁 Thư mục: {folder_path}")
            print("-" * 60)

            deleted_count = 0
            failed_count = 0
            results = []

            for idx, filename in enumerate(all_files, 1):
                # Tạo document_id từ tên file (bỏ extension)
                filename_without_ext = os.path.splitext(filename)[0]
                # Sanitize giống như khi embed
                document_id = re.sub(r'[^\w\-_.]', '_', filename_without_ext)

                print(f"\n[{idx}/{total_files}] Xóa: {filename}")
                print(f"   Document ID: {document_id}")

                success, message = self.delete_document_embeddings(document_id)

                if success:
                    deleted_count += 1
                    results.append({
                        "filename": filename,
                        "document_id": document_id,
                        "status": "deleted",
                        "message": message
                    })
                else:
                    failed_count += 1
                    results.append({
                        "filename": filename,
                        "document_id": document_id,
                        "status": "failed",
                        "error": message
                    })

                # Delay nhỏ giữa các request
                if idx < total_files:
                    time.sleep(0.3)

            print("\n" + "=" * 60)
            print("KẾT QUẢ XÓA EMBEDDINGS")
            print("=" * 60)
            print(f"✓ Tổng số files: {total_files}")
            print(f"✓ Xóa thành công: {deleted_count}")
            print(f"✗ Xóa thất bại: {failed_count}")
            print(f"📈 Tỷ lệ thành công: {(deleted_count / total_files * 100):.1f}%")

            return {
                "success": True,
                "total": total_files,
                "deleted": deleted_count,
                "failed": failed_count,
                "success_rate": round(deleted_count / total_files * 100, 1) if total_files > 0 else 0,
                "results": results
            }

        except Exception as e:
            print(f"❌ Lỗi khi xóa embeddings: {e}")
            return {
                "success": False,
                "error": str(e),
                "total": 0,
                "deleted": 0,
                "failed": 0
            }

        """
        Xóa tất cả embeddings của các documents đã được crawl từ CSV

        Args:
            csv_file_path: Đường dẫn đến file CSV (mặc định: output_dir/danh_sach_van_ban.csv)

        Returns:
            dict: Thống kê kết quả xóa
        """
        if csv_file_path is None:
            csv_file_path = os.path.join(self.output_dir, 'danh_sach_van_ban.csv')

        if not os.path.exists(csv_file_path):
            print(f"❌ Không tìm thấy file CSV: {csv_file_path}")
            return {
                "success": False,
                "error": "CSV file not found",
                "total": 0,
                "deleted": 0,
                "failed": 0
            }

        try:
            print("=" * 60)
            print("BẮT ĐẦU XÓA TẤT CẢ EMBEDDINGS")
            print("=" * 60)

            # Đọc CSV
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

            # Lọc các documents đã có embedding
            if 'embedding_status' in df.columns:
                embedded_docs = df[df['embedding_status'] == 'success'].copy()
            else:
                # Nếu không có cột embedding_status, lấy tất cả documents có so_ky_hieu
                embedded_docs = df[df['so_ky_hieu'].notna()].copy()

            total_docs = len(embedded_docs)

            if total_docs == 0:
                print("⚠️  Không tìm thấy document nào cần xóa")
                return {
                    "success": True,
                    "total": 0,
                    "deleted": 0,
                    "failed": 0,
                    "message": "No documents to delete"
                }

            print(f"\n📊 Tìm thấy {total_docs} documents cần xóa embeddings")
            print("-" * 60)

            deleted_count = 0
            failed_count = 0
            results = []

            for idx, row in embedded_docs.iterrows():
                so_ky_hieu = row['so_ky_hieu']

                # Tạo document_id từ số ký hiệu (giống như khi embed)
                document_id = re.sub(r'[^\w\-_.]', '_', str(so_ky_hieu))

                print(f"\n[{idx + 1}/{total_docs}] Xóa: {so_ky_hieu}")

                success, message = self.delete_document_embeddings(document_id)

                if success:
                    deleted_count += 1
                    results.append({
                        "document_id": document_id,
                        "so_ky_hieu": so_ky_hieu,
                        "status": "deleted",
                        "message": message
                    })
                else:
                    failed_count += 1
                    results.append({
                        "document_id": document_id,
                        "so_ky_hieu": so_ky_hieu,
                        "status": "failed",
                        "error": message
                    })

                # Delay nhỏ giữa các request
                if idx < total_docs - 1:
                    time.sleep(0.5)

            print("\n" + "=" * 60)
            print("KẾT QUẢ XÓA EMBEDDINGS")
            print("=" * 60)
            print(f"✓ Tổng số documents: {total_docs}")
            print(f"✓ Xóa thành công: {deleted_count}")
            print(f"✗ Xóa thất bại: {failed_count}")
            print(f"📈 Tỷ lệ thành công: {(deleted_count / total_docs * 100):.1f}%")

            return {
                "success": True,
                "total": total_docs,
                "deleted": deleted_count,
                "failed": failed_count,
                "success_rate": round(deleted_count / total_docs * 100, 1) if total_docs > 0 else 0,
                "results": results
            }

        except Exception as e:
            print(f"❌ Lỗi khi xóa embeddings: {e}")
            return {
                "success": False,
                "error": str(e),
                "total": 0,
                "deleted": 0,
                "failed": 0
            }

    def crawl_and_embed(self, max_pages=None, download_files=True, auto_embed=True, delay=1):
        """
        Crawl văn bản và tự động embedding vào vector DB

        Args:
            max_pages: Số trang tối đa cần crawl (None = tất cả)
            download_files: Có tải file không
            auto_embed: Có tự động embedding không
            delay: Thời gian delay giữa các request (giây)
        """
        print("=" * 60)
        print("BẮT ĐẦU CRAWL & EMBEDDING DX.GOV.VN")
        print("=" * 60)

        # Lấy tổng số trang
        print("\n[1/5] Đang xác định tổng số trang...")
        total_pages = self.get_total_pages()
        print(f"✓ Tổng số trang tìm thấy: {total_pages}")

        if max_pages:
            total_pages = min(total_pages, max_pages)
            print(f"✓ Giới hạn crawl: {total_pages} trang")

        all_documents = []

        # Crawl từng trang
        print(f"\n[2/5] Bắt đầu crawl {total_pages} trang...")
        print("-" * 60)

        for page in range(1, total_pages + 1):
            documents = self.crawl_page(page)
            all_documents.extend(documents)

            if page % 10 == 0 or page == total_pages:
                print(f"   Progress: {page}/{total_pages} trang ({len(all_documents)} văn bản)")

            if page < total_pages:
                time.sleep(delay)

        print("-" * 60)
        print(f"✓ Hoàn thành crawl: {len(all_documents)} văn bản từ {total_pages} trang")

        # Lưu dữ liệu
        print(f"\n[3/5] Đang lưu dữ liệu...")
        df = pd.DataFrame(all_documents)
        csv_path = os.path.join(self.output_dir, 'danh_sach_van_ban.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ Đã lưu danh sách vào: {csv_path}")

        # Tải xuống và embedding file
        if download_files:
            print(f"\n[4/5] Bắt đầu tải file...")
            print("-" * 60)

            downloaded = 0
            failed = 0
            skipped = 0
            embedded = 0
            embed_failed = 0
            total_with_link = len([d for d in all_documents if d['download_link']])

            for idx, doc in enumerate(all_documents, 1):
                if doc['download_link']:
                    print(f"\n📥 [{idx}/{len(all_documents)}] {doc['so_ky_hieu']}")

                    # Tải file
                    success, saved_filename, filepath = self.download_file(
                        doc['download_link'],
                        doc['so_ky_hieu']
                    )

                    if success:
                        downloaded += 1
                        doc['saved_file'] = saved_filename
                        doc['file_path'] = filepath

                        # Auto embedding nếu được bật
                        if auto_embed and filepath:
                            # Tạo document_id từ số ký hiệu
                            document_id = re.sub(r'[^\w\-_.]', '_', doc['so_ky_hieu'])

                            # Process document
                            markdown_content, error = self.process_document_api(filepath)

                            if markdown_content:
                                # Embed markdown
                                embed_success, embed_result = self.embed_markdown_api(
                                    markdown_content,
                                    document_id
                                )

                                if embed_success:
                                    embedded += 1
                                    doc['embedding_status'] = 'success'
                                    doc['embeddings_count'] = embed_result.get('stored_count', 0)
                                    doc['document_id'] = document_id  # Lưu document_id
                                else:
                                    embed_failed += 1
                                    doc['embedding_status'] = 'failed'
                                    doc['embedding_error'] = str(embed_result)
                            else:
                                embed_failed += 1
                                doc['embedding_status'] = 'process_failed'
                                doc['embedding_error'] = str(error)

                        # Progress
                        if downloaded % 5 == 0:
                            print(f"\n📊 Progress: {downloaded}/{total_with_link} files downloaded")
                            if auto_embed:
                                print(f"   🔗 Embedded: {embedded}/{downloaded}")
                            time.sleep(delay)
                    else:
                        failed += 1
                        doc['saved_file'] = None
                        doc['embedding_status'] = 'download_failed'
                else:
                    skipped += 1
                    doc['saved_file'] = None
                    doc['embedding_status'] = 'no_link'

            print("-" * 60)
            print(f"\n=== THỐNG KÊ TẢI FILE ===")
            print(f"✓ Tải thành công: {downloaded}/{total_with_link}")
            print(f"✗ Tải thất bại: {failed}")
            print(f"⊘ Không có link: {skipped}")

            if auto_embed:
                print(f"\n=== THỐNG KÊ EMBEDDING ===")
                print(f"✓ Embedding thành công: {embedded}/{downloaded}")
                print(f"✗ Embedding thất bại: {embed_failed}")
                print(f"📈 Tỷ lệ thành công: {(embedded / downloaded * 100):.1f}%" if downloaded > 0 else "0%")

            # Cập nhật lại CSV với thông tin file đã lưu
            df = pd.DataFrame(all_documents)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ Đã cập nhật: {csv_path}")

        print("\n" + "=" * 60)
        print("HOÀN THÀNH!")
        print("=" * 60)

        return df


# Sử dụng
if __name__ == "__main__":
    # Khởi tạo crawler với API URL mới
    crawler = DXGovCrawlerWithEmbedding(
        output_dir="van_ban_downloads"
    )

    # ===== MENU LỰA CHỌN =====
    print("\n" + "=" * 60)
    print("🤖 DX.GOV.VN CRAWLER & EMBEDDING TOOL")
    print("=" * 60)
    print("Chọn chức năng:")
    print("1. Crawl và Embed văn bản (KHUYẾN NGHỊ)")
    print("2. Xóa tất cả embeddings từ thư mục file")
    print("3. Xóa tất cả embeddings từ CSV")
    print("4. Xóa một document cụ thể")
    print("0. Thoát")
    print("=" * 60)

    choice = input("\nNhập lựa chọn của bạn (0-4): ").strip()

    if choice == "1":
        # CRAWL VÀ EMBED
        print("\n" + "=" * 60)
        print("🚀 CRAWL VÀ EMBED VĂN BẢN")
        print("=" * 60)

        # Hỏi số trang
        max_pages_input = input("\nSố trang muốn crawl (Enter = tất cả, hoặc nhập số): ").strip()
        max_pages = None
        if max_pages_input and max_pages_input.isdigit():
            max_pages = int(max_pages_input)
            print(f"✓ Sẽ crawl {max_pages} trang")
        else:
            print("✓ Sẽ crawl TẤT CẢ trang (có thể mất nhiều thời gian)")

        # Xác nhận
        confirm = input("\nBắt đầu crawl? (y/n): ").strip().lower()

        if confirm == 'y' or confirm == 'yes':
            print("\n🚀 Bắt đầu crawl và embedding...")
            df = crawler.crawl_and_embed(
                max_pages=max_pages,
                download_files=True,
                auto_embed=True,
                delay=2
            )

            print("\n✅ HOÀN THÀNH!")
            print(f"📊 Tổng số văn bản: {len(df)}")
            if 'embedding_status' in df.columns:
                print(f"✓ Embedded thành công: {(df['embedding_status'] == 'success').sum()}")
        else:
            print("\n❌ Đã hủy!")

    elif choice == "2":
        # XÓA TỪ THỦ MỤC FILE
        print("\n⚠️  CẢNH BÁO: Bạn sắp xóa TẤT CẢ embeddings từ các file đã download!")
        print("=" * 60)

        folder_path = crawler.output_dir
        if not os.path.exists(folder_path):
            print(f"❌ Không tìm thấy thư mục: {folder_path}")
        else:
            # Đếm số file
            supported_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
            file_count = sum(len([f for f in os.listdir(folder_path)
                                  if f.lower().endswith(ext)])
                             for ext in supported_extensions)

            print(f"📁 Thư mục: {folder_path}")
            print(f"📊 Tìm thấy: {file_count} files")

            if file_count == 0:
                print("\n⚠️  Không có file nào để xóa!")
            else:
                confirm = input("\nXác nhận xóa embeddings của tất cả file? (y/n): ").strip().lower()

                if confirm == 'y' or confirm == 'yes':
                    print("\n🚀 Bắt đầu xóa embeddings...")
                    result = crawler.delete_embeddings_from_folder()

                    print("\n" + "=" * 60)
                    print("📊 KẾT QUẢ CUỐI CÙNG")
                    print("=" * 60)
                    print(f"   Tổng files: {result['total']}")
                    print(f"   ✓ Xóa thành công: {result['deleted']}")
                    print(f"   ✗ Xóa thất bại: {result['failed']}")
                    if result.get('success_rate'):
                        print(f"   📈 Tỷ lệ thành công: {result['success_rate']}%")
                else:
                    print("\n❌ Đã hủy!")

    elif choice == "3":
        # XÓA TỪ CSV
        print("\n⚠️  CẢNH BÁO: Bạn sắp xóa TẤT CẢ embeddings từ CSV đã crawl!")
        print("=" * 60)

        csv_path = os.path.join(crawler.output_dir, 'danh_sach_van_ban.csv')
        if not os.path.exists(csv_path):
            print(f"❌ Không tìm thấy file CSV: {csv_path}")
        else:
            confirm = input("\nXác nhận xóa? (y/n): ").strip().lower()

            if confirm == 'y' or confirm == 'yes':
                print("\n🚀 Bắt đầu xóa embeddings...")
                result = crawler.delete_all_embeddings_from_csv()

                print("\n" + "=" * 60)
                print("📊 KẾT QUẢ CUỐI CÙNG")
                print("=" * 60)
                print(f"   Tổng documents: {result['total']}")
                print(f"   ✓ Xóa thành công: {result['deleted']}")
                print(f"   ✗ Xóa thất bại: {result['failed']}")
                if result.get('success_rate'):
                    print(f"   📈 Tỷ lệ thành công: {result['success_rate']}%")
            else:
                print("\n❌ Đã hủy!")

    elif choice == "4":
        # XÓA MỘT DOCUMENT CỤ THỂ
        print("\n🗑️  XÓA MỘT DOCUMENT CỤ THỂ")
        print("=" * 60)

        document_id = input("Nhập document_id cần xóa: ").strip()

        if document_id:
            confirm = input(f"\nXác nhận xóa document '{document_id}'? (y/n): ").strip().lower()

            if confirm == 'y' or confirm == 'yes':
                success, message = crawler.delete_document_embeddings(document_id)
                if success:
                    print(f"\n✅ {message}")
                else:
                    print(f"\n❌ {message}")
            else:
                print("\n❌ Đã hủy!")
        else:
            print("\n❌ Document ID không hợp lệ!")

    elif choice == "0":
        print("\n👋 Tạm biệt!")

    else:
        print("\n❌ Lựa chọn không hợp lệ!")

    # ===== SỬ DỤNG TRỰC TIẾP KHÔNG QUA MENU =====

    # Cách 1: Crawl và embed trực tiếp
    # df = crawler.crawl_and_embed(
    #     max_pages=5,  # Số trang muốn crawl (None = tất cả)
    #     download_files=True,
    #     auto_embed=True,
    #     delay=2
    # )

    # Cách 2: Xóa từ thư mục
    # result = crawler.delete_embeddings_from_folder()

    # Cách 3: Xóa từ CSV
    # result = crawler.delete_all_embeddings_from_csv()

    # Cách 4: Xóa document cụ thể
    # success, message = crawler.delete_document_embeddings("123_2024_QD-UBND")
