#include "Inu_generalized_hough_transform.h"

using namespace std;
using namespace cv;

inu_generalized_hough_transform::inu_generalized_hough_transform()
{

}

inu_generalized_hough_transform::~inu_generalized_hough_transform()
{

}

vector<int> inu_generalized_hough_transform::get_object_labels(vector<vector<Vec2i>>& indices, int thresh)
{
	vector<int> obj;
	indices.clear();

	for (int i = 0; i < m_nlabels; ++i)
	{
		for (int j = 0; j < m_nx; ++j)
		{
			for (int k = 0; k < m_ny; ++k)
			{
				for (int m = 0; m < m_ns; ++m)
				{
					for (int n = 0; n < m_no; ++n)
					{
						if (m_bin[i][j][k][m][n] >= thresh)
						{
							obj.push_back(m_label[i]);
							vector<Vec2i> index;
							for (int s = 0; s < m_index[i][j][k][m][n].size(); ++s)
								index.push_back(m_index[i][j][k][m][n][s]);
							indices.push_back(index);
						}
					}
				}
			}
		}
	}

	return obj;
}

void inu_generalized_hough_transform::vote(int l, float x, float y, float s, float o, Vec2i& ind)
{
	int xpos = (int)((x - m_x[0]) / m_step_x + 0.5);
	if (xpos < 0 || xpos >= m_nx)
		return;

	int ypos = (int)((y - m_y[0]) / m_step_y + 0.5);
	if (ypos < 0 || ypos >= m_ny)
		return;

	int spos = (int)(log(s / m_s[0]) / log(m_step_s) + 0.5);
	if (spos < 0 || spos >= m_ns)
		return;

	int opos = (int)((o - m_o[0]) / m_step_o + 0.5);
	if (opos < 0)
		return;
	if (opos >= m_no)
		opos -= m_no;

	m_bin[l][xpos][ypos][spos][opos] += 1;
	m_index[l][xpos][ypos][spos][opos].push_back(ind);
}

void inu_generalized_hough_transform::init(vector<int>& labels, float min_x, float max_x, float min_y, float max_y, float min_scale, float max_scale, float res_x, float res_y, float res_s, float res_ori)
{
	// object labels
	m_nlabels = (int)labels.size();
	m_label = labels;

	m_nx = (int)(ceil((max_x - min_x) / res_x)+1);

	if (m_nx > 1)
		m_step_x = max((max_x - min_x), 0.0f) / max((double)(m_nx - 1), 1.0);
	else
		m_step_x = 1.0;

	m_x.push_back(min_x);
	for (int i = 1; i < m_nx; ++i)
		m_x.push_back((float)(min_x + m_step_x*i));

	m_ny = (int)(ceil((max_y - min_y) / res_y) + 1);
	if (m_ny > 1)
		m_step_y = max((max_y - min_y), 0.0f) / max((double)(m_ny - 1), 1.0);
	else
		m_step_y = 1.0;

	m_y.push_back(min_y);
	for (int i = 1; i < m_ny; ++i)
		m_y.push_back((float)(min_y + m_step_y*i));

	m_ns = (int)(ceil(log(max_scale / min_scale) / log(res_s)) + 1);
	
	if (m_ns > 1)
		m_step_s = exp(max(log(max_scale / (double)min_scale), 0.0) / max((m_ns - 1), 1));
	else
		m_step_s = 1.1;

	m_s.push_back(min_scale);
	for (int i = 1; i < m_ns; ++i)
		m_s.push_back((float)(min_scale*pow(m_step_s, i)));

	if (res_ori > 45.0f)
		res_ori = 45.0f;
	m_no = (int)((360.0f-res_ori) / res_ori+0.5)+1;
	m_step_o = 360.0/m_no;

	m_o.push_back(0);
	for (int i = 1; i < m_no; ++i)
		m_o.push_back((float)(m_step_o*i));

	m_bin.clear();
	m_index.clear();
	m_bin.resize(m_nlabels);
	m_index.resize(m_nlabels);
	for (int i = 0; i < m_nlabels; ++i)
	{
		m_bin[i].resize(m_nx);
		m_index[i].resize(m_nx);
		for (int j = 0; j < m_nx; ++j)
		{
			m_bin[i][j].resize(m_ny);
			m_index[i][j].resize(m_ny);
			for (int k = 0; k < m_ny; ++k)
			{
				m_bin[i][j][k].resize(m_ns);
				m_index[i][j][k].resize(m_ns);
				for (int m=0;m<m_ns;++m)
				{
					m_bin[i][j][k][m].resize(m_no);
					m_index[i][j][k][m].resize(m_no);
					for (int n = 0; n < m_no; ++n)
						m_bin[i][j][k][m][n] = 0;
				}
			}
		}
	}
}