from sentence_transformers import SentenceTransformer, util
import torch
from bs4 import BeautifulSoup
from typing import Dict, Any
import time

class JobMatchAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with GPU if available"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)
        print("Model loaded successfully")

    def clean_html(self, html_text: str) -> str:
        """Clean HTML tags from text"""
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text(separator=' ').strip()

    def analyze_match(self, cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze match between CV and job post"""
        start_time = time.time()
        
        try:
            # Extract CV information
            cv_info = cv_data['Data'][0]
            cv_skills = set()
            cv_skills.update(cv_info['professional'].get('technical_skills', []))
            cv_skills.update(cv_info['professional'].get('non_technical_skills', []))
            cv_skills.update(cv_info['professional'].get('tools', []))
            
            # Get years of experience
            cv_experience = float(cv_info['professional']['experience'][0]['years'][0])
            
            # Extract job requirements
            required_skills = set(skill.lower() for skill in job_data['skillSets'])
            required_experience = float(job_data['experienceRequired'])
            
            # Calculate skill match
            cv_skills_lower = set(skill.lower() for skill in cv_skills)
            matching_skills = required_skills.intersection(cv_skills_lower)
            missing_skills = required_skills - cv_skills_lower
            extra_skills = cv_skills_lower - required_skills
            
            skill_match_score = len(matching_skills) / len(required_skills) if required_skills else 0
            
            # Calculate experience match
            exp_match_score = min(cv_experience / required_experience, 1.0) if required_experience > 0 else 1.0
            
            # Prepare texts for semantic similarity
            job_text = f"""
            {job_data['jobTitle']}
            {self.clean_html(job_data['jobDescription'])}
            {self.clean_html(job_data['qualificationRequired'])}
            Required Skills: {', '.join(job_data['skillSets'])}
            """
            
            cv_text = f"""
            Role: {cv_info['professional']['experience'][0]['role'][0]}
            Skills: {', '.join(cv_skills)}
            Experience: {cv_experience} years
            Project Experience: {' '.join(cv_info['professional']['experience'][0].get('project_experience', []))}
            Education: {', '.join([qual for edu in cv_info['professional']['education'] for qual in edu['qualification']])}
            """
            
            # Calculate semantic similarity using GPU
            with torch.no_grad():
                job_embedding = self.model.encode(job_text, convert_to_tensor=True)
                cv_embedding = self.model.encode(cv_text, convert_to_tensor=True)
                
                if self.device.type == 'cuda':
                    job_embedding = job_embedding.cuda()
                    cv_embedding = cv_embedding.cuda()
                
                similarity = util.cos_sim(job_embedding, cv_embedding).cpu().numpy()[0][0]
            
            # Calculate final score
            weights = {
                'skills': 0.4,
                'experience': 0.3,
                'content': 0.3
            }
            
            final_score = (
                skill_match_score * weights['skills'] +
                exp_match_score * weights['experience'] +
                float(similarity) * weights['content']
            ) * 100
            
            # Prepare result
            result = {
                "success": True,
                "processing_time": round(time.time() - start_time, 2),
                "device_used": str(self.device),
                "match_details": {
                    "job_id": job_data.get('id'),
                    "job_title": job_data.get('jobTitle'),
                    "candidate_name": cv_info['personal']['name'][0],
                    "candidate_email": cv_info['personal']['email'][0],
                    "scores": {
                        "overall_match": round(final_score, 2),
                        "skill_match": round(skill_match_score * 100, 2),
                        "experience_match": round(exp_match_score * 100, 2),
                        "content_similarity": round(float(similarity) * 100, 2)
                    },
                    "skill_analysis": {
                        "matching_skills": sorted(list(matching_skills)),
                        "missing_skills": sorted(list(missing_skills)),
                        "additional_skills": sorted(list(extra_skills))
                    },
                    "experience_analysis": {
                        "required_years": required_experience,
                        "candidate_years": cv_experience,
                        "meets_requirement": cv_experience >= required_experience
                    },
                    "recommendation": {
                        "category": "Strong Match" if final_score >= 75 else "Moderate Match" if final_score >= 50 else "Low Match",
                        "action": "Highly Recommended for Interview" if final_score >= 75 else 
                                "Consider for Interview" if final_score >= 50 else 
                                "May Need Additional Skills/Experience"
                    }
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": round(time.time() - start_time, 2),
                "device_used": str(self.device)
            }