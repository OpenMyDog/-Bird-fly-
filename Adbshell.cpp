//#include "StdAfx.h"

#include "Adbshell.h"
#include <iostream>

using namespace std;

DWORD __stdcall ThreadAdbshellProc(void *pVoid) {
	if (pVoid != NULL) {
		CAdbshell *pShell = (CAdbshell *)pVoid;
		pShell->Loop();
	}
	return 0;
}


CAdbshell::CAdbshell(void) {
	m_hEvent = NULL;
	//m_hProcess = NULL;
}

CAdbshell::~CAdbshell(void) {
}

//����adb shell������ɹ������#״̬������ʹ��RunCmdִ������
BOOL CAdbshell::Start() {
	m_strOutput.Empty();
	SECURITY_ATTRIBUTES sat;
	STARTUPINFO startupinfo;
	PROCESS_INFORMATION pinfo;

	sat.nLength = sizeof(SECURITY_ATTRIBUTES);
	sat.bInheritHandle = TRUE;
	sat.lpSecurityDescriptor = NULL;
	if (CreatePipe(&m_hReadPipe, &m_hWritePipe, &sat, NULL) == FALSE) {
		return FALSE;
	}
	if (CreatePipe(&m_hReadPipe2, &m_hWritePipe2, &sat, NULL) == FALSE) {
		return FALSE;
	}
	startupinfo.cb = sizeof(STARTUPINFO);
	GetStartupInfo(&startupinfo);
	startupinfo.hStdError = m_hWritePipe;
	startupinfo.hStdOutput = m_hWritePipe;
	startupinfo.hStdInput = m_hReadPipe2;
	startupinfo.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
	startupinfo.wShowWindow = SW_HIDE;
	if (CreateProcess(NULL, "adb.exe shell", NULL, NULL, TRUE, 0, NULL, NULL, &startupinfo, &pinfo) == FALSE) {
		//DWORD dwError = GetLastError();
		cout << "adb���ߵ���ʧ��" << endl;
		return FALSE;
	}
	CloseHandle(m_hWritePipe);
	CloseHandle(m_hReadPipe2);
	CloseHandle(pinfo.hThread);
	CloseHandle(pinfo.hProcess);
	//m_hProcess = pinfo.hProcess;

	DWORD dwThread = FALSE;
	hThread = CreateThread(NULL, 0, ThreadAdbshellProc, this, 0, &dwThread);//�����������������Ҫ����Ҫ��Ҫ
	if (hThread == NULL) {
		return FALSE;
	}

	m_hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

	return TRUE;
}

//�������������adb shell״̬�µ��������������ж�������\n�ָ�
BOOL CAdbshell::RunCmd(const CString&strCmdline) {
	BOOL bSuccess = TRUE;
	CString strOneCmd;
	int nPos1 = 0;
	int nPos2 = 0;
	DWORD dwWrite = 0;

	if (strCmdline.GetLength() < 2) {
		return FALSE;
	}
	//�ȵȴ��̺߳���׼����
	WaitForSingleObject(m_hEvent, INFINITE);

	while (TRUE) {
		nPos1 = nPos2;
		nPos2 = strCmdline.Find('\n', nPos1);
		if (nPos2 == -1) {
			nPos2 = strCmdline.GetLength();
		}
		strOneCmd = strCmdline.Mid(nPos1, nPos2 - nPos1).Trim();

		//���������Ϊ2
		if (strOneCmd.GetLength() >= 2) {
			strOneCmd += "\r\n";
			//ִ��
			bSuccess = WriteFile(m_hWritePipe2, strOneCmd, strOneCmd.GetLength(), &dwWrite, NULL);
		}

		++nPos2;
		if (nPos2 >= strCmdline.GetLength()) {
			break;
		}
	}

	return bSuccess;
}

//�˳�shell����״̬���رս��̡�����ͨ��TerminateProcess��ʽ������������ж�ȡ��ȫ�����
BOOL CAdbshell::Stop() {
	RunCmd("exit");
	WaitForSingleObject(hThread, INFINITE);
	return TRUE;
	//if ( m_hProcess!=NULL ) {
	//	TerminateProcess(m_hProcess, -1);
	//	CloseHandle(m_hProcess);
	//	m_hProcess = NULL;
	//}

	//return TRUE;
}

//��ȡ������,����ǰ����ص���Stop�Ƚ���
CString CAdbshell::GetOutput() {
	WaitForSingleObject(hThread, INFINITE);

	//����exitҪȥ��
	int nPos1 = 0;
	//int nPos2 = 0;

	nPos1 = m_strOutput.Find("# exit");
	if (nPos1 == -1) {
		nPos1 = m_strOutput.Find("$ exit");
	}
	if (nPos1 != -1) {
		m_strOutput = m_strOutput.Left(nPos1);
		nPos1 = m_strOutput.ReverseFind('\n');
		m_strOutput = m_strOutput.Left(nPos1);
	}

	return m_strOutput;
}

BOOL CAdbshell::Loop() {
	CHAR buffer[1024] = { 0 };
	DWORD dwRead = 0;

	while (TRUE) {
		RtlZeroMemory(buffer, _countof(buffer));
		SetEvent(m_hEvent);
		if (ReadFile(m_hReadPipe, buffer, _countof(buffer), &dwRead, NULL) == FALSE) {
			break;
		} else {
			m_strOutput += buffer;
			if (StrStrA(buffer, "# exit") != NULL) {
				break;
			}
		}
	}
	CloseHandle(m_hReadPipe);
	CloseHandle(m_hWritePipe2);

	CloseHandle(m_hEvent);
	m_hEvent = NULL;
	CloseHandle(hThread);
	hThread = NULL;

	return TRUE;
}