"""
MedlinePlus Health Topics XML to Neo4j Knowledge Graph Converter

This script parses MedlinePlus Health Topics XML files and creates a 
comprehensive knowledge graph in Neo4j with nodes for health topics,
symptoms, tests, organizations, and their relationships.
"""

import xml.etree.ElementTree as ET
from neo4j import GraphDatabase
from typing import Dict, List, Set, Optional
import re
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

class MedlinePlusNeo4jConverter:
    """Convert MedlinePlus XML to Neo4j knowledge graph"""
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI (e.g., 'bolt://localhost:7687')
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def create_constraints_and_indexes(self):
        """Create uniqueness constraints and indexes for better performance"""
        constraints = [
            "CREATE CONSTRAINT health_topic_id IF NOT EXISTS FOR (ht:HealthTopic) REQUIRE ht.id IS UNIQUE",
            "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT test_name IF NOT EXISTS FOR (t:Test) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT group_name IF NOT EXISTS FOR (g:Group) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT mesh_id IF NOT EXISTS FOR (m:MeshDescriptor) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT synonym_name IF NOT EXISTS FOR (s:Synonym) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:InformationCategory) REQUIRE c.name IS UNIQUE",
            "CREATE INDEX site_url IF NOT EXISTS FOR (s:Site) ON (s.url)"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created: {constraint.split()[1]}")
                except Exception as e:
                    print(f"Constraint/Index already exists or error: {e}")
    
    def extract_symptoms_from_text(self, text: str) -> List[str]:
        """
        Extract symptoms from the full summary text
        
        Args:
            text: Full summary text containing symptoms
            
        Returns:
            List of extracted symptom strings
        """
        symptoms = []
        
        # Look for common symptom patterns
        symptom_patterns = [
            r'<li>\s*([^<]+)\s*</li>',  # List items often contain symptoms
            r'symptoms[^:]*:\s*([^.]+)',  # After "symptoms:"
            r'can include\s+([^.]+)',  # After "can include"
            r'may have\s+([^.]+)',  # After "may have"
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the text
                symptom = re.sub(r'<[^>]+>', '', match)  # Remove HTML tags
                symptom = symptom.strip()
                
                # Split by common delimiters
                sub_symptoms = re.split(r'[,;]|\s+and\s+|\s+or\s+', symptom)
                for s in sub_symptoms:
                    s = s.strip()
                    if s and len(s) > 3 and len(s) < 100:  # Filter out noise
                        symptoms.append(s)
        
        return list(set(symptoms))  # Remove duplicates
    
    def extract_tests_from_text(self, text: str) -> List[str]:
        """
        Extract medical tests from the full summary text
        
        Args:
            text: Full summary text
            
        Returns:
            List of test names
        """
        tests = []
        
        # Common test keywords
        test_keywords = [
            r'(CT\s+scan)', r'(MRI)', r'(X-ray)', r'(ultrasound)',
            r'(biopsy)', r'(blood\s+test)', r'(hearing\s+test)',
            r'(ear\s+exam)', r'(scan)', r'(examination)',
            r'(computed\s+tomography)', r'(magnetic\s+resonance)',
            r'(radiosurgery)', r'(imaging)'
        ]
        
        for pattern in test_keywords:
            matches = re.findall(pattern, text, re.IGNORECASE)
            tests.extend([m if isinstance(m, str) else m[0] for m in matches])
        
        return list(set([t.strip() for t in tests if t]))
    
    def extract_diagnostics_from_text(self, text: str) -> List[str]:
        """
        Extract diagnostic procedures from the health topic record
        
        Args:
            text: Full summary text
        Returns:
            List of diagnostic procedure names
        """


    
    def parse_health_topic(self, health_topic_elem: ET.Element) -> Dict:
        """
        Parse a single health-topic XML element
        
        Args:
            health_topic_elem: XML element for health-topic
            
        Returns:
            Dictionary containing parsed health topic data
        """
        data = {
            'id': health_topic_elem.get('id'),
            'title': health_topic_elem.get('title'),
            'url': health_topic_elem.get('url'),
            'language': health_topic_elem.get('language', 'English'),
            'date_created': health_topic_elem.get('date-created'),
            'synonyms': [],
            'full_summary': '',
            'groups': [],
            'mesh_descriptors': [],
            'related_topics': [],
            'sites': [],
            'primary_institute': None,
            'symptoms': [],
            'tests': []
        }
        
        # Parse also-called (synonyms)
        for synonym in health_topic_elem.findall('also-called'):
            if synonym.text:
                data['synonyms'].append(synonym.text.strip())
        
        # Parse full summary
        full_summary = health_topic_elem.find('full-summary')
        if full_summary is not None:
            # Get all text including from child elements
            data['full_summary'] = ET.tostring(full_summary, encoding='unicode', method='text')
            
            # Extract symptoms and tests from summary
            data['symptoms'] = self.extract_symptoms_from_text(data['full_summary'])
            data['tests'] = self.extract_tests_from_text(data['full_summary'])
        
        # Parse groups
        for group in health_topic_elem.findall('group'):
            data['groups'].append({
                'name': group.text.strip() if group.text else '',
                'url': group.get('url'),
                'id': group.get('id')
            })
        
        # Parse MeSH headings
        for mesh in health_topic_elem.findall('.//mesh-heading/descriptor'):
            data['mesh_descriptors'].append({
                'id': mesh.get('id'),
                'name': mesh.text.strip() if mesh.text else ''
            })
        
        # Parse primary institute
        institute = health_topic_elem.find('primary-institute')
        if institute is not None:
            data['primary_institute'] = {
                'name': institute.text.strip() if institute.text else '',
                'url': institute.get('url')
            }
        
        # Parse related topics
        for related in health_topic_elem.findall('related-topic'):
            data['related_topics'].append({
                'id': related.get('id'),
                'title': related.text.strip() if related.text else '',
                'url': related.get('url')
            })
        
        # Parse sites
        for site in health_topic_elem.findall('site'):
            site_data = {
                'title': site.get('title'),
                'url': site.get('url'),
                'organizations': [],
                'categories': []
            }
            
            # Get organizations
            for org in site.findall('organization'):
                if org.text:
                    site_data['organizations'].append(org.text.strip())
            
            # Get information categories
            for category in site.findall('information-category'):
                if category.text:
                    site_data['categories'].append(category.text.strip())
            
            # for ic in site.findall('information-category'):
            #     if ic.text.strip() == 'Treatments and Therapies':
            #         treatments_sites.append({
            #             'title': site.get('title'),
            #             'url': site.get('url'),
            #             'organization': site.find('organization').text if site.find('organization') is not None else None
            #         })
            #     if ic.text.strip() == 'Diagnosis and Tests':
            #         diagnosis_sites.append({
            #             'title': site.get('title'),
            #             'url': site.get('url'),
            #             'organization': site.find('organization').text if site.find('organization') is not None else None
            #         })

            data['sites'].append(site_data)
        
        return data
    
    def create_health_topic_node(self, tx, data: Dict):
        """Create HealthTopic node"""
        query = """
        MERGE (ht:HealthTopic {id: $id})
        SET ht.title = $title,
            ht.url = $url,
            ht.language = $language,
            ht.date_created = $date_created,
            ht.full_summary = $full_summary,
            ht.updated_at = datetime()
        RETURN ht
        """
        tx.run(query, 
               id=data['id'],
               title=data['title'],
               url=data['url'],
               language=data['language'],
               date_created=data['date_created'],
               full_summary=data['full_summary'])
    
    def create_symptoms(self, tx, health_topic_id: str, symptoms: List[str]):
        """Create Symptom nodes and relationships"""
        for symptom in symptoms:
            query = """
            MATCH (ht:HealthTopic {id: $ht_id})
            MERGE (s:Symptom {name: $symptom_name})
            MERGE (ht)-[:HAS_SYMPTOM]->(s)
            """
            tx.run(query, ht_id=health_topic_id, symptom_name=symptom)
    
    def create_tests(self, tx, health_topic_id: str, tests: List[str]):
        """Create Test nodes and relationships"""
        for test in tests:
            query = """
            MATCH (ht:HealthTopic {id: $ht_id})
            MERGE (t:Test {name: $test_name})
            MERGE (ht)-[:HAS_TEST]->(t)
            """
            tx.run(query, ht_id=health_topic_id, test_name=test)
    
    def create_groups(self, tx, health_topic_id: str, groups: List[Dict]):
        """Create Group nodes and relationships"""
        for group in groups:
            if group['name']:
                query = """
                MATCH (ht:HealthTopic {id: $ht_id})
                MERGE (g:Group {name: $group_name})
                SET g.url = $url,
                    g.id = $group_id
                MERGE (ht)-[:IN_GROUP]->(g)
                """
                tx.run(query, 
                       ht_id=health_topic_id,
                       group_name=group['name'],
                       url=group['url'],
                       group_id=group['id'])
    
    def create_synonyms(self, tx, health_topic_id: str, synonyms: List[str]):
        """Create Synonym nodes and relationships"""
        for synonym in synonyms:
            query = """
            MATCH (ht:HealthTopic {id: $ht_id})
            MERGE (s:Synonym {name: $synonym_name})
            MERGE (ht)-[:HAS_SYNONYM]->(s)
            """
            tx.run(query, ht_id=health_topic_id, synonym_name=synonym)
    
    def create_mesh_descriptors(self, tx, health_topic_id: str, 
                               mesh_descriptors: List[Dict]):
        """Create MeshDescriptor nodes and relationships"""
        for mesh in mesh_descriptors:
            if mesh['id'] and mesh['name']:
                query = """
                MATCH (ht:HealthTopic {id: $ht_id})
                MERGE (m:MeshDescriptor {id: $mesh_id})
                SET m.name = $mesh_name
                MERGE (ht)-[:HAS_MESH]->(m)
                """
                tx.run(query, 
                       ht_id=health_topic_id,
                       mesh_id=mesh['id'],
                       mesh_name=mesh['name'])
    
    def create_related_topics(self, tx, health_topic_id: str, 
                             related_topics: List[Dict]):
        """Create relationships to related health topics"""
        for related in related_topics:
            if related['id']:
                query = """
                MATCH (ht1:HealthTopic {id: $ht_id1})
                MERGE (ht2:HealthTopic {id: $ht_id2})
                SET ht2.title = $related_title,
                    ht2.url = $related_url
                MERGE (ht1)-[:RELATED_TO]->(ht2)
                """
                tx.run(query,
                       ht_id1=health_topic_id,
                       ht_id2=related['id'],
                       related_title=related['title'],
                       related_url=related['url'])
    
    def create_sites(self, tx, health_topic_id: str, sites: List[Dict]):
        """Create Site, Organization, and InformationCategory nodes"""
        for site in sites:
            # Create Site node
            site_query = """
            MATCH (ht:HealthTopic {id: $ht_id})
            MERGE (s:Site {url: $url})
            SET s.title = $title
            MERGE (ht)-[:HAS_RESOURCE]->(s)
            """
            tx.run(site_query,
                   ht_id=health_topic_id,
                   url=site['url'],
                   title=site['title'])
            
            # Create Organizations and relationships
            for org_name in site['organizations']:
                org_query = """
                MATCH (s:Site {url: $url})
                MERGE (o:Organization {name: $org_name})
                MERGE (s)-[:PROVIDED_BY]->(o)
                """
                tx.run(org_query, url=site['url'], org_name=org_name)
            
            # Create Information Categories and relationships
            for category in site['categories']:
                cat_query = """
                MATCH (s:Site {url: $url})
                MERGE (c:InformationCategory {name: $category})
                MERGE (s)-[:HAS_CATEGORY]->(c)
                """
                tx.run(cat_query, url=site['url'], category=category)
    
    def create_primary_institute(self, tx, health_topic_id: str, 
                                institute: Optional[Dict]):
        """Create relationship to primary institute organization"""
        if institute and institute['name']:
            query = """
            MATCH (ht:HealthTopic {id: $ht_id})
            MERGE (o:Organization {name: $org_name})
            SET o.url = $url
            MERGE (ht)-[:PRIMARY_INSTITUTE]->(o)
            """
            tx.run(query,
                   ht_id=health_topic_id,
                   org_name=institute['name'],
                   url=institute['url'])
    
    def create_test_organization_relationships(self, tx, health_topic_id: str, 
                                               sites: List[Dict]):
        """
        Create relationships between Tests and Organizations
        by inferring from site information categories
        """
        for site in sites:
            # If site has "Diagnosis and Tests" category, link tests to orgs
            if any('test' in cat.lower() or 'diagnosis' in cat.lower() 
                   for cat in site['categories']):
                for org_name in site['organizations']:
                    query = """
                    MATCH (ht:HealthTopic {id: $ht_id})-[:HAS_TEST]->(t:Test)
                    MATCH (o:Organization {name: $org_name})
                    MERGE (t)-[:PROVIDED_BY]->(o)
                    """
                    tx.run(query, ht_id=health_topic_id, org_name=org_name)
    
    def process_health_topic(self, health_topic_data: Dict):
        """Process a single health topic and create all nodes/relationships"""
        with self.driver.session() as session:
            # Create HealthTopic node
            session.execute_write(self.create_health_topic_node, health_topic_data)
            
            # Create Symptoms
            if health_topic_data['symptoms']:
                session.execute_write(self.create_symptoms, 
                                    health_topic_data['id'],
                                    health_topic_data['symptoms'])
            
            # Create Tests
            if health_topic_data['tests']:
                session.execute_write(self.create_tests,
                                    health_topic_data['id'],
                                    health_topic_data['tests'])
            
            # Create Groups
            if health_topic_data['groups']:
                session.execute_write(self.create_groups,
                                    health_topic_data['id'],
                                    health_topic_data['groups'])
            
            # Create Synonyms
            if health_topic_data['synonyms']:
                session.execute_write(self.create_synonyms,
                                    health_topic_data['id'],
                                    health_topic_data['synonyms'])
            
            # Create MeSH Descriptors
            if health_topic_data['mesh_descriptors']:
                session.execute_write(self.create_mesh_descriptors,
                                    health_topic_data['id'],
                                    health_topic_data['mesh_descriptors'])
            
            # Create Related Topics
            if health_topic_data['related_topics']:
                session.execute_write(self.create_related_topics,
                                    health_topic_data['id'],
                                    health_topic_data['related_topics'])
            
            # Create Sites, Organizations, and Categories
            if health_topic_data['sites']:
                session.execute_write(self.create_sites,
                                    health_topic_data['id'],
                                    health_topic_data['sites'])
                
                # Create Test-Organization relationships
                session.execute_write(self.create_test_organization_relationships,
                                    health_topic_data['id'],
                                    health_topic_data['sites'])
            
            # Create Primary Institute
            if health_topic_data['primary_institute']:
                session.execute_write(self.create_primary_institute,
                                    health_topic_data['id'],
                                    health_topic_data['primary_institute'])
    
    def process_xml_file(self, xml_file_path: str):
        """
        Process entire XML file and create knowledge graph
        
        Args:
            xml_file_path: Path to MedlinePlus XML file
        """
        print(f"Parsing XML file: {xml_file_path}")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Find all health-topic elements
        health_topics = root.findall('.//health-topic')
        total = len(health_topics)
        
        print(f"Found {total} health topics")
        print("Creating constraints and indexes...")
        self.create_constraints_and_indexes()
        
        print("Processing health topics...")
        for idx, health_topic_elem in enumerate(health_topics, 1):
            try:
                data = self.parse_health_topic(health_topic_elem)
                self.process_health_topic(data)
                
                if idx % 10 == 0:
                    print(f"Processed {idx}/{total} health topics...")
            except Exception as e:
                print(f"Error processing health topic {idx}: {e}")
                continue
        
        print(f"\nCompleted! Processed {total} health topics.")
        self.print_statistics()
    
    def print_statistics(self):
        """Print statistics about the created knowledge graph"""
        with self.driver.session() as session:
            stats_query = """
            MATCH (ht:HealthTopic) WITH count(ht) as health_topics
            MATCH (s:Symptom) WITH health_topics, count(s) as symptoms
            MATCH (t:Test) WITH health_topics, symptoms, count(t) as tests
            MATCH (g:Group) WITH health_topics, symptoms, tests, count(g) as groups
            MATCH (o:Organization) WITH health_topics, symptoms, tests, groups, count(o) as orgs
            MATCH (m:MeshDescriptor) WITH health_topics, symptoms, tests, groups, orgs, count(m) as mesh
            MATCH (site:Site) WITH health_topics, symptoms, tests, groups, orgs, mesh, count(site) as sites
            RETURN health_topics, symptoms, tests, groups, orgs, mesh, sites
            """
            result = session.run(stats_query).single()
            
            print("\n" + "="*50)
            print("KNOWLEDGE GRAPH STATISTICS")
            print("="*50)
            print(f"Health Topics:         {result['health_topics']}")
            print(f"Symptoms:              {result['symptoms']}")
            print(f"Tests:                 {result['tests']}")
            print(f"Groups:                {result['groups']}")
            print(f"Organizations:         {result['orgs']}")
            print(f"MeSH Descriptors:      {result['mesh']}")
            print(f"Sites:                 {result['sites']}")
            print("="*50)


# ==================== Usage Example ====================

def main():
    """Example usage of the converter"""
    
    # Neo4j connection details
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # Path to MedlinePlus XML file
    XML_FILE_PATH = "data/mplus_topics_2025-11-19.xml"
    
    # Initialize converter
    converter = MedlinePlusNeo4jConverter(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    try:
        # Optional: Clear existing data (be careful!)
        # converter.clear_database()
        
        # Process XML file and create knowledge graph
        converter.process_xml_file(XML_FILE_PATH)
        
        print("\n✅ Knowledge graph created successfully!")
        
        # Example: Query the graph
        with converter.driver.session() as session:
            # Find health topics with specific symptoms
            query = """
            MATCH (ht:HealthTopic)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE s.name CONTAINS 'hearing'
            RETURN ht.title, collect(s.name) as symptoms
            LIMIT 5
            """
            results = session.run(query)
            
            print("\nExample Query - Health topics related to hearing:")
            for record in results:
                print(f"  • {record['ht.title']}: {record['symptoms']}")
        
    finally:
        converter.close()


if __name__ == "__main__":
    main()